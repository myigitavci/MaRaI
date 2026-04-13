import os
import torch
from torch import nn
from torch.nn import functional as F
import errno
import nibabel as nib
from torchvision import utils
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import fsspec
from torchvision import transforms as T

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def reparameterize_logit(logit):
    import warnings
    warnings.filterwarnings('ignore', message='.*Mixed memory format inputs detected.*')
    beta = F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)
    return beta


def save_image(images, file_name):
    image_save = torch.cat([image[:4, [0], ...].cpu() for image in images], dim=0)
    image_save = utils.make_grid(tensor=image_save, nrow=4, normalize=False, range=(0, 1)).detach().numpy()[0, ...]
    image_save = nib.Nifti1Image(image_save.transpose(1, 0), np.eye(4))
    nib.save(image_save, file_name)


def dropout_contrasts(available_contrast_id, contrast_id_to_drop=None):
    """
    Randomly dropout contrasts for MR-Styler training.

    ==INPUTS==j
    * available_contrast_id: torch.Tensor (batch_size, num_contrasts)
        Indicates the availability of each MR contrast. 1: if available, 0: if unavailable.

    * contrast_id_to_drop: torch.Tensor (batch_size, num_contrasts)
        If provided, indicates the contrast indexes forced to drop. Default: None

    ==OUTPUTS==
    * contrast_id_after_dropout: torch.Tensor (batch_size, num_contrasts)
        Some available contrasts will be randomly dropped out (as if they are unavailable).
        However, each sample will have at least one contrast available.
    """
    batch_size = available_contrast_id.shape[0]
    if contrast_id_to_drop is not None:
        available_contrast_id = available_contrast_id - contrast_id_to_drop
    contrast_id_after_dropout = available_contrast_id.clone()
    for i in range(batch_size):
        available_contrast_ids_per_subject = (available_contrast_id[i] == 1).nonzero(as_tuple=False).squeeze(1)
        num_available_contrasts = available_contrast_ids_per_subject.numel()
        if num_available_contrasts > 1:
            num_contrast_to_drop = torch.randperm(num_available_contrasts - 1)[0]
            contrast_ids_to_drop = torch.randperm(num_available_contrasts)[:num_contrast_to_drop]
            contrast_ids_to_drop = available_contrast_ids_per_subject[contrast_ids_to_drop]
            contrast_id_after_dropout[i, contrast_ids_to_drop] = 0.0
    return contrast_id_after_dropout

class PerceptualLoss2(nn.Module):
    def __init__(self, vgg_model, layers=(4, 9, 16), layer_weights=None):
        """
        layers: tuple of VGG feature layer indices to extract features from
        layer_weights: weights for each layer loss (defaults to equal)
        """
        super().__init__()
        
        # Freeze VGG parameters
        for param in vgg_model.parameters():
            param.requires_grad = False
        
        # Store chosen feature slices
        self.vgg_slices = nn.ModuleList([
            nn.Sequential(*list(vgg_model.children())[:l+1]).eval()
            for l in layers
        ])
        
        # If no weights provided, use equal weighting
        if layer_weights is None:
            layer_weights = [1.0] * len(layers)
        self.layer_weights = layer_weights


    def forward(self, x, y):
        # Repeat grayscale to RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)

        # Extract features & compute L1 loss per layer
        total_loss = 0.0
        for slice_net, weight in zip(self.vgg_slices, self.layer_weights):
            fx = slice_net(x)
            fy = slice_net(y)
            total_loss += weight * F.l1_loss(fx, fy)

        return total_loss
    
class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg_model.children())[:13]).eval()

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(x), self.vgg(y))


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature

    def forward(self, query_feature, positive_feature, negative_feature):
        B, C, N = query_feature.shape

        l_positive = (query_feature * positive_feature).sum(dim=1)[:, :, None]
        l_negative = torch.bmm(query_feature.permute(0, 2, 1), negative_feature)

        logits = torch.cat((l_positive, l_negative), dim=2) / self.temperature

        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * N, dtype=torch.long).to(query_feature.device)
        return self.ce_loss(predictions, targets).mean()


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        kld_loss = -0.5 * logvar + 0.5 * (torch.exp(logvar) + torch.pow(mu, 2)) - 0.5
        return kld_loss


def divide_into_batches(in_tensor, num_batches):
    batch_size = in_tensor.shape[0] // num_batches
    remainder = in_tensor.shape[0] % num_batches
    batches = []

    current_start = 0
    for i in range(num_batches):
        current_end = current_start + batch_size
        if remainder:
            current_end += 1
            remainder -= 1
        batches.append(in_tensor[current_start:current_end, ...])
        current_start = current_end
    return batches


def normalize_intensity(image):
    thresh = np.percentile(image.flatten(), 95)
    image = image / (thresh + 1e-5)
    image = np.clip(image, a_min=0.0, a_max=5.0)
    return image, thresh


def zero_pad(image, image_dim=256):
    [n_row, n_col, n_slc] = image.shape
    image_padded = np.zeros((image_dim, image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2] = image
    return image_padded

def zero_pad2d(image, image_dim=256):
    [n_row, n_col] = image.shape
    image_padded = np.zeros((image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2] = image
    return image_padded


def crop(image, n_row, n_col, n_slc):
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2]

def crop2d(image, n_row, n_col):
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2]
def save_training_visualization(src, tgt, rec, beta, mask, out_path, title=None):
    import matplotlib.pyplot as plt
    import numpy as np

    imgs = [src, tgt, mask, rec, beta]
    imgs = [img.detach().cpu().numpy() if hasattr(img, 'detach') else img for img in imgs]
    imgs = [img.squeeze() for img in imgs]  # remove batch/channel dim if present

    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for ax, img, label in zip(axes, imgs, ['Source', 'Target', 'Mask', 'Recon', 'Beta']):
        # Normalize beta values to [0, 1] range for proper visualization
        if label == 'Beta':
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:  # Avoid division by zero
                img = (img - img_min) / (img_max - img_min)
        
        if img.ndim == 2:  # (H, W)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        elif img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
            ax.imshow(np.transpose(img, (1, 2, 0)))
        else:  # already (H, W, C)
            ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
def pt_load(file_path, map_location=None):
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out

def init_distributed_training(local_rank, world_size):
    """Initialize distributed training"""
    if local_rank >= 0:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
        torch.cuda.set_device(local_rank)
        return True
    return False
def get_clip_augmentation():
    """Data augmentation for CLIP training (matching CLIPStyler)"""
    
    
    return T.Compose([
        T.RandomResizedCrop(
            224,
            scale=(0.7, 1.0),  # Reduced crop range to minimize smoothing
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Reduced transformation
        T.RandomHorizontalFlip(),
        # T.GaussianBlur(kernel_size=3),  # Removed Gaussian blur to preserve sharpness
        T.Resize(224)
    ])

def generate_experiment_name(beta_dim, use_beta, use_patchifier, use_contrast_feat, use_adversarial=False, use_perceptual=False, discriminator_update_freq=5, discriminator_channels=32, discriminator_layers=5, discriminator_lr_scale=0.5, batch_size=None, lr=None, max_slices_per_step=None, gpu_id=0, prefix=None, free_text=None, beta_type='weighted', image_contrast='text', perceptual_type=None, weights=None):
    """Generate a descriptive experiment name based on key parameters"""
    components = []
    
    # Model architecture components
    if use_beta:
        components.append(f"beta{beta_dim}")
    else:
        components.append("nobeta")
    
    if use_patchifier:
        components.append("patch")
    else:
        components.append("nopatch")
    
    # Contrast feature method
    if use_contrast_feat == 'adain':
        components.append("adain")
    elif use_contrast_feat == 'adapter':
        components.append("adapter")
    elif use_contrast_feat == 'attention':
        components.append("attn")
    elif use_contrast_feat == 'enhanced':
        components.append("enhanced")
    elif use_contrast_feat == 'multiscale':
        components.append("multiscale")
    elif use_contrast_feat == 'enhancedv2':
        components.append("enhancedv2")
    elif use_contrast_feat == 'enhancedv3':
        components.append("enhancedv3")
    else:
        components.append("basic")
    
    # Beta type
    if beta_type == 'weighted':
        components.append("weighted")
    elif beta_type == 'discrete':
        components.append("discrete")
    elif beta_type == 'simple':
        components.append("simple")
    elif beta_type == 'minimal':
        components.append("minimal")
    elif beta_type == 'old':
        components.append("old")

    # Adversarial training
    if use_adversarial:
        components.append(f"adv{discriminator_update_freq}")
        components.append(f"disc_ch{discriminator_channels}")
        components.append(f"disc_l{discriminator_layers}")
        components.append(f"disc_lr{discriminator_lr_scale}")
    else:
        components.append("noadv")
    
    # Perceptual loss
    if use_perceptual:
        if perceptual_type is None:
            components.append("perceptual")
        else:
            components.append(f"per_{perceptual_type}")
    else:
        components.append("noperceptual")
    
    if image_contrast == 'image':
        components.append("image_contrast")
    elif image_contrast == 'text':
        components.append("text_contrast")
    elif image_contrast == 'both':
        components.append("both_contrast")
    
    # Training parameters (if provided)
    if batch_size is not None:
        components.append(f"bs{batch_size}")
    
    if lr is not None:
        # Format learning rate nicely (e.g., 1e-4 -> 1e4, 0.001 -> 1e3)
        if lr >= 0.001:
            lr_str = f"{lr:.0e}".replace("e+0", "e").replace("e-0", "e-")
        else:
            lr_str = f"{lr:.0e}".replace("e-0", "e-")
        components.append(f"lr{lr_str}")
    
    if max_slices_per_step is not None:
        components.append(f"slice{max_slices_per_step}")
    
    # GPU info
    components.append(f"gpu{gpu_id}")
    
    # Loss weights (optional, compact)
    if weights is not None:
        def fmt(v):
            try:
                if v == int(v):
                    return str(int(v))
            except Exception:
                pass
            return ("%g" % float(v)).replace('.', 'p')
        components.append(
            "w" + 
            f"r{fmt(weights.get('w_rec', 1.0))}" +
            f"_c{fmt(weights.get('w_clip', 0.0))}" +
            f"_l2{fmt(weights.get('w_clip_l2', 0.0))}" +
            f"_ci{fmt(weights.get('w_clip_img', 0.0))}" +
            f"_p{fmt(weights.get('w_per', 0.0))}" +
            f"_b{fmt(weights.get('w_beta', 0.0))}" +
            f"_a{fmt(weights.get('w_adv', 0.0))}"
        )
    
    # Join all components
    experiment_name = "_".join(components)
    
    # Add free text if provided
    if free_text is not None and free_text.strip():
        # Clean the free text (remove special characters that might cause issues)
        clean_text = free_text.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
        experiment_name = f"{experiment_name}_{clean_text}"
    
    return experiment_name
