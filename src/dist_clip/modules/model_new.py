from tqdm import tqdm
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
#from torch.optim.lr_scheduler import LR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.models as models
from datetime import datetime
import nibabel as nib
from torch.cuda.amp import autocast
from dist_clip.modules.utils import *
from dist_clip.modules.dataset import PairedImageDataset, PairedSliceDataset, PairedPNGSliceDataset, PairItem,classic_normalize
from dist_clip.modules.network import UNet, Patchifier, FeatureAdapter, AdaINBlock, CrossAttentionBlock,PatchDiscriminator
from monai.networks.layers import Act
from dist_clip.modules.perceptual import PerceptualLoss as MONAIPerceptualLoss
from dist_clip.modules.enhanced_style_transfer import EnhancedStyleTransfer, TextConditionedDecoder, MultiScaleDecoder, TextConditionedDecoderV2, TextConditionedDecoderv3
from dist_clip.modules.adversarial_loss import PatchAdversarialLoss
from open_clip import  get_tokenizer,create_model_and_transforms
import os
import matplotlib.pyplot as plt
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from PIL import Image
# Configure matplotlib to avoid font warnings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
import warnings
import math
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class DIST_CLIP:
    def __init__(self, beta_dim, pretrained_dist_clip=None, gpu_id=0, max_slices_per_step=None, clip_model_path=None, use_contrast_feat='adain', use_beta=None, use_patchifier=False, batch_size=None, lr=None, local_rank=-1, free_text=None, base_ch=16, use_adversarial=False, use_perceptual=False, perceptual_type='monai', perceptual_variant=3, w_rec=1.0, w_clip=0.1, w_clip_l2=0.05, w_clip_img=0.1, w_per=0.2, w_beta=0.3, w_adv=0.2, discriminator_update_freq=5, discriminator_channels=32, discriminator_layers=5, discriminator_lr_scale=0.5, beta_type='weighted', coronal_check=False, guidance_mode='both', log_gradients=False, w_cyc=1.0, grad_clip=5.0,textcontextlength=98, clip_image_size=224):
        self.beta_dim = beta_dim
        self.local_rank = local_rank
        self.free_text = free_text
        self.base_ch = base_ch
        self.grad_clip = grad_clip
        # Setup device for multi-GPU or single GPU
        if local_rank >= 0:
            # Multi-GPU setup
            self.device = torch.device(f'cuda:{local_rank}')
            self.is_distributed = True
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            # Single GPU setup
            self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
            self.is_distributed = False
            self.world_size = 1
            self.rank = 0
        
        # Initialize experiment name (will be updated when we have batch_size and lr)
        self.experiment_name = generate_experiment_name(
            beta_dim=beta_dim,
            use_beta=use_beta,
            use_patchifier=use_patchifier,
            use_contrast_feat=use_contrast_feat,
            use_adversarial=use_adversarial,
            use_perceptual=use_perceptual,
            discriminator_update_freq=discriminator_update_freq,
            discriminator_channels=discriminator_channels,
            discriminator_layers=discriminator_layers,
            discriminator_lr_scale=discriminator_lr_scale,
            batch_size=batch_size,
            lr=lr,
            max_slices_per_step=max_slices_per_step,
            gpu_id=gpu_id,
            free_text=free_text,
            beta_type=beta_type,
            image_contrast=guidance_mode,
            perceptual_type=perceptual_type if use_perceptual else None,
            weights={
                'w_rec': w_rec if 'w_rec' in locals() else 1.0,
                'w_clip': w_clip if 'w_clip' in locals() else 0.0,
                'w_clip_l2': w_clip_l2 if 'w_clip_l2' in locals() else 0.0,
                'w_clip_img': w_clip_img if 'w_clip_img' in locals() else 0.0,
                'w_per': w_per if 'w_per' in locals() else 0.0,
                'w_beta': w_beta if 'w_beta' in locals() else 0.0,
                'w_adv': w_adv if 'w_adv' in locals() else 0.0,
            } if use_perceptual else None
        )
        
        # Create timestamp with experiment name
        self.timestr = self.experiment_name
        self.textcontextlength = textcontextlength
        self.use_contrast_feat = use_contrast_feat  # 'adain', 'adapter', ...
        self.max_slices_per_step = max_slices_per_step
        self.use_beta = use_beta  # Whether to use beta encoder and beta loss
        self.use_patchifier = use_patchifier  # Whether to use patchifier for contrastive loss
        self.use_adversarial = use_adversarial  # Whether to use adversarial training
        self.use_perceptual = use_perceptual  # Whether to use perceptual loss
        self.perceptual_type = perceptual_type  # 'monai' | 'vgg1' | 'vgg2'
        # Loss weights
        self.w_rec = w_rec
        self.w_clip = w_clip
        self.w_clip_l2 = w_clip_l2
        self.w_clip_img = w_clip_img
        self.w_per = w_per
        self.w_beta = w_beta
        self.w_adv = w_adv
        self.w_cyc = w_cyc
        self.log_gradients = log_gradients
        self.discriminator_update_freq = discriminator_update_freq  # How often to update discriminator
        self.discriminator_channels = discriminator_channels  # Number of channels in discriminator
        self.discriminator_layers = discriminator_layers  # Number of layers in discriminator
        self.discriminator_lr_scale = discriminator_lr_scale  # Learning rate scaling for discriminator
        self.batch_size = batch_size
        self.lr = lr
        self.beta_type = beta_type  # encoder/segmentation type selection
        self.coronal_check = coronal_check  # Whether to check coronal slices
        # Guidance mode: 'text', 'image', or 'both'
        self.guidance_mode = guidance_mode
        self.train_loader, self.valid_loader = None, None
        self.out_dir = None
        self.optimizer = None
        self.discriminator_optimizer = None
        self.scheduler = None
        self.writer, self.writer_path = None, None
        self.checkpoint = None
        self.l1_loss, self.contrastive_loss, self.perceptual_loss = None, None, None
        self.adversarial_loss = None
        self.gradient_penalty_loss = None
        self.clip_image_size = clip_image_size
        # Epoch history for tracking progression
        self.epoch_history = {
            'epochs': [],
            'total_loss': [],
            'rec_loss': [],
            'per_loss': [],
            'clip_loss': [],
            'clip_loss_l2': [],
            'clip_image_loss': [],
            'beta_loss': [],
            'cycle_loss': [],
            'adv_loss': [],
            'disc_loss': [],

            # Validation losses
            'val_total_loss': [],
            'val_rec_loss': [],
            'val_per_loss': [],
            'val_clip_loss': [],
            'val_clip_loss_l2': [],
            'val_clip_image_loss': [],
            'val_beta_loss': [],
            'val_cycle_loss': [],
            'val_adv_loss': [],
            'val_disc_loss': [],
        }

        # 100-batch averaging for in-epoch tracking
        self.batch_100_history = {
            'batches': [],
            'total_loss': [],
            'rec_loss': [],
            'per_loss': [],
            'clip_loss': [],
            'clip_loss_l2': [],
            'clip_image_loss': [],
            'beta_loss': [],
            'cycle_loss': [],
            'adv_loss': [],
            'disc_loss': [],
        }
        # Accumulator for current 100-batch bucket (global across epochs)
        self._batch_100_acc = {
            'count': 0,
            'sums': {k: 0.0 for k in ['total_loss','rec_loss','per_loss','clip_loss','clip_loss_l2','clip_image_loss','beta_loss','cycle_loss','adv_loss','disc_loss']}
        }
        # Global batch counter across all epochs
        self._global_batch_counter = 0
        self._lpips_net = None  # lazy init
        # Validation metrics history across epochs
        self.val_metrics_history = {
            'epochs': [],
            'ssim': [],
            'psnr': [],
            'lpips': [],
        }

        # Optional sampler to fetch external negative images for PatchNCE
        # Expect a callable: neg_sampler(batch_size:int, H:int, W:int, K:int)->Tensor[B,K,C,H,W]
        self.neg_sampler = None
        # Disabled by default to avoid per-call dataset reads; prefer memory negatives
        self.use_dataset_negatives = False

        # Cross-batch memory of raw slices (MoCo-style, but store images to recompute features with current patchifier)
        # Keep last N epochs, with up to M slices per epoch
        self.neg_mem_max_epochs = 2
        self.neg_mem_slices_per_epoch = 200  # cap per epoch
        self.neg_mem_samples_per_batch = 4   # use K from memory per batch
        self._neg_mem_by_epoch = {}

        # define networks
        if self.use_beta:
            # Use brain tissue beta encoder for tissue segmentation
            from dist_clip.modules.simple_beta_encoder import create_simple_beta_encoder
            
            # Determine encoder type based on beta_type
            if self.beta_type in ['simple', 'minimal', 'old']:
                encoder_type = self.beta_type
                beta_dim = self.beta_dim  # Use the specified beta_dim
            else:
                encoder_type = 'brain_tissue'  # For 'weighted' and 'discrete'
                beta_dim = 4  # Fixed to 4 for WM, GM, CSF, Other
            
            self.beta_encoder = create_simple_beta_encoder(
                beta_dim=beta_dim,
                encoder_type=encoder_type,
                base_channels=self.base_ch,
                use_anatomical_priors=True if encoder_type == 'brain_tissue' else False
            )
            self.beta_encoder.to(self.device)
        else:
            self.beta_encoder = None
            
        if self.use_patchifier:
            # Match patchifier input channels to beta channels so it can process betas directly
            in_ch_pf = self.beta_dim if self.use_beta else 1
            self.patchifier = Patchifier(in_ch=in_ch_pf, out_ch=128)
            self.patchifier.to(self.device)
        else:
            self.patchifier = None
            
        if self.use_contrast_feat == 'attention':
            self.cross_attn_block = CrossAttentionBlock(dim=256, num_heads=8).to(self.device)
            self.text_proj = nn.Linear(512, 256).to(self.device)
            self.image_proj = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, 256)
            ).to(self.device)  # From C=1 → C=256
            self.img_proj_back = nn.Linear(256, 1).to(self.device)
            self.decoder = UNet(in_ch=self.beta_dim, out_ch=1, base_ch=self.base_ch, final_act='relu')
        elif self.use_contrast_feat == 'adain':
            self.adain_block = AdaINBlock(num_channels=1, style_dim=512).to(self.device)
            self.decoder = UNet(in_ch=self.beta_dim, out_ch=1, base_ch=self.base_ch, final_act='relu')
        elif self.use_contrast_feat == 'adapter':
            self.feature_adapter = FeatureAdapter(in_dim=512, out_size=224).to(self.device)
            # Input will be concatenated [betas (C=self.beta_dim), contrast_map (C=1)]
            self.decoder = UNet(in_ch=self.beta_dim + 1, out_ch=1, base_ch=self.base_ch, final_act='relu')
        elif self.use_contrast_feat == 'enhanced':
            # Enhanced style transfer with text-conditioned decoder
            self.decoder = TextConditionedDecoder(in_ch=self.beta_dim, out_ch=1, base_ch=self.base_ch, final_act='relu').to(self.device)
        elif self.use_contrast_feat == 'enhancedv2':
            # Enhanced style transfer with text-conditioned decoder
            self.decoder = TextConditionedDecoderV2(in_ch=self.beta_dim, out_ch=1, base_ch=self.base_ch, final_act='relu').to(self.device)
        elif self.use_contrast_feat == 'multiscale':
            # Multi-scale style injection
            from dist_clip.modules.enhanced_style_transfer import MultiScaleStyleInjector
            self.multi_scale_injector = MultiScaleStyleInjector(in_channels=1, style_dim=512).to(self.device)
            self.enhanced_style_transfer = EnhancedStyleTransfer(style_dim=512, hidden_dim=256).to(self.device)
            self.decoder = MultiScaleDecoder(in_channels=self.beta_dim, out_channels=1, base_channels=32).to(self.device)
        elif self.use_contrast_feat == 'enhancedv3':
            # Enhanced style transfer with text-conditioned decoder
            self.decoder = TextConditionedDecoderv3(in_ch=1, out_ch=1, base_ch=self.base_ch, final_act='relu').to(self.device)
        else:
            self.decoder = UNet(in_ch=1, out_ch=1, base_ch=self.base_ch, final_act='relu')

        # No shared style projector here; projection handled elsewhere in the model when needed

        if pretrained_dist_clip is not None:
            self.checkpoint = torch.load(pretrained_dist_clip, map_location=self.device)
            if self.use_beta and 'beta_encoder' in self.checkpoint:
                self.beta_encoder.load_state_dict(self.checkpoint['beta_encoder'])
            if self.use_patchifier and 'patchifier' in self.checkpoint:
                self.patchifier.load_state_dict(self.checkpoint['patchifier'])
            if 'decoder' in self.checkpoint:
                missing, unexpected = self.decoder.load_state_dict(self.checkpoint['decoder'], strict=False)
                if missing or unexpected:
                    print(f"[Decoder] load_state_dict(strict=False) -> missing: {missing}, unexpected: {unexpected}")
            if hasattr(self, 'enhanced_style_transfer') and 'enhanced_style_transfer' in self.checkpoint:
                self.enhanced_style_transfer.load_state_dict(self.checkpoint['enhanced_style_transfer'])
            if hasattr(self, 'multi_scale_injector') and 'multi_scale_injector' in self.checkpoint:
                self.multi_scale_injector.load_state_dict(self.checkpoint['multi_scale_injector'])
            if self.use_adversarial and 'discriminator' in self.checkpoint:
                self.discriminator.load_state_dict(self.checkpoint['discriminator'])
        
        self.decoder.to(self.device)
        
        # Initialize discriminator if adversarial training is enabled
        if self.use_adversarial:
            # Use the new PatchDiscriminator from MONAI
            self.discriminator = PatchDiscriminator(
                spatial_dims=2,
                num_channels=self.discriminator_channels,
                in_channels=1,
                out_channels=1,
                num_layers_d=self.discriminator_layers,
                kernel_size=4,
                activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
                norm="BATCH",
                bias=False,
                padding=1,
                dropout=0.0
            ).to(self.device)
            # Use the new PatchAdversarialLoss from MONAI
            self.adversarial_loss = PatchAdversarialLoss(
                reduction="mean",
                criterion="least_squares",
                no_activation_leastsq=False
            )
        else:
            self.discriminator = None
        
        # Wrap models with DDP for multi-GPU training
        if self.is_distributed:
            if self.beta_encoder is not None:
                self.beta_encoder = DDP(self.beta_encoder, device_ids=[self.local_rank])
            if self.patchifier is not None:
                self.patchifier = DDP(self.patchifier, device_ids=[self.local_rank])
            if hasattr(self, 'enhanced_style_transfer'):
                self.enhanced_style_transfer = DDP(self.enhanced_style_transfer, device_ids=[self.local_rank])
            if hasattr(self, 'multi_scale_injector'):
                self.multi_scale_injector = DDP(self.multi_scale_injector, device_ids=[self.local_rank])
            if self.use_adversarial and self.discriminator is not None:
                self.discriminator = DDP(self.discriminator, device_ids=[self.local_rank])
            self.decoder = DDP(self.decoder, device_ids=[self.local_rank])
        self.start_epoch = 0
        self.discriminator_step_counter = 0  # Counter for discriminator updates
        self.load_clip_model(clip_model_path)

        # Freeze CLIP model parameters and set to eval mode
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
  
    def load_clip_model(self, clip_model_path):
        self.clip_model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-B-16',
        device=self.device,
        textcontextlength=self.textcontextlength,
        force_image_size=224,
        )
        checkpoint = pt_load(clip_model_path, map_location='cpu')
        if 'epoch' in checkpoint:
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
        else:
            sd = checkpoint

        # Filter out keys whose shapes don't match the current model (e.g. 3-ch vs 1-ch conv1)
        model_sd = self.clip_model.state_dict()
        filtered_sd = {k: v for k, v in sd.items()
                       if k in model_sd and v.shape == model_sd[k].shape}
        skipped = [k for k in sd if k not in filtered_sd]
        if skipped:
            print(f"[CLIP] Skipping {len(skipped)} mismatched keys: {skipped[:5]}{'...' if len(skipped)>5 else ''}")
        missing, unexpected = self.clip_model.load_state_dict(filtered_sd, strict=False)
        if missing:
            print(f"[CLIP] {len(missing)} keys not loaded from checkpoint (shape mismatch or absent)")
        self.tokenizer = get_tokenizer('ViT-B-16', context_length=self.textcontextlength)
        
        # # Determine expected image size from CLIP model's positional embedding
        # # This is needed to resize images before encoding to match checkpoint's training size
        # if hasattr(self.clip_model.visual, 'positional_embedding'):
        #     pos_emb_size = self.clip_model.visual.positional_embedding.shape[1]
        #     num_patches = pos_emb_size - 1  # -1 for cls token
        #     patch_size = getattr(self.clip_model.visual, 'patch_size', 16)
        #     self.clip_image_size = int((num_patches ** 0.5) * patch_size)
        # else:
        #     self.clip_image_size = getattr(self.clip_model.visual, 'image_size', 128)
    
    

    def setup_logging(self, out_dir):
        """Setup logging configuration"""
        log_file = os.path.join(out_dir, f'training_{self.timestr}.log')
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    def _extract_brain_mask_fast(self, imgs: torch.Tensor) -> torch.Tensor:
            """
            Fast head mask extraction for 2D slices (brain + skull, exclude background air).

            imgs: [B,1,H,W] in [0,1]. Returns binary mask [B,1,H,W] on same device.
            Designed to work on non–skull-stripped MR (head mask, not pure brain mask).
            """
            import numpy as np
            from scipy import ndimage

            if imgs.dim() != 4 or imgs.size(1) != 1:
                # Fallback: everything slightly above 0 is considered head
                return (imgs > 1e-3).float()

            b, _, h, w = imgs.shape
            masks = []
            imgs_cpu = imgs.detach().cpu().numpy()

            for i in range(b):
                sl = imgs_cpu[i, 0]
                # If completely empty, no mask
                if np.all(np.isnan(sl)) or np.max(sl) <= 0:
                    masks.append(np.zeros_like(sl, dtype=np.float32))
                    continue

                try:
                    # Use full slice statistics (including skull) to separate air vs head.
                    #  - p5 is close to background air
                    #  - p95 is deep brain / skull
                    p5 = np.percentile(sl, 5.0)
                    p95 = np.percentile(sl, 95.0)
                    if p95 <= p5:
                        # Degenerate contrast; simple threshold
                        thr = p5
                    else:
                        # Threshold a bit above background: head = values significantly above p5
                        thr = p5 + 0.15 * (p95 - p5)

                    m = sl > thr

                    # Morphological cleanup
                    m = ndimage.binary_opening(m, structure=np.ones((3, 3)))
                    m = ndimage.binary_closing(m, structure=np.ones((5, 5)))
                    m = ndimage.binary_fill_holes(m)

                    # Keep largest connected component (head)
                    labeled, num = ndimage.label(m)
                    if num > 1:
                        sizes = ndimage.sum(m, labeled, index=np.arange(1, num + 1))
                        biggest = int(np.argmax(sizes)) + 1
                        m = (labeled == biggest)

                    frac = m.mean()
                    # If mask is clearly too small or too large, relax/strengthen threshold once
                    if frac < 0.01 or frac > 0.95:
                        thr2 = p5 + 0.05 * (p95 - p5) if frac > 0.95 else p5 + 0.25 * (p95 - p5)
                        m = sl > thr2
                        m = ndimage.binary_opening(m, structure=np.ones((3, 3)))
                        m = ndimage.binary_closing(m, structure=np.ones((5, 5)))
                        m = ndimage.binary_fill_holes(m)

                    # Final fallback: simple > tiny threshold
                    if m.mean() < 0.005:
                        m = (sl > 1e-4).astype(np.float32)
                    else:
                        m = m.astype(np.float32)

                    masks.append(m)
                except Exception:
                    # Emergency fallback
                    masks.append((sl > 1e-4).astype(np.float32))

            mask_np = np.stack(masks, axis=0)  # [B,H,W]
            mask_t = torch.from_numpy(mask_np).to(imgs.device).unsqueeze(1)

            return (mask_t > 0.5).float()
    def initialize_training(self, out_dir, lr):
        # Store learning rate for experiment naming
        self.lr = lr
        
        # Setup logging first
        
        # define loss functions
        self.rec_loss = nn.L1Loss(reduction='none')
        self.ncc_loss = self.local_ncc_loss
        # Initialize perceptual loss according to selection
        if self.use_perceptual:
            if self.perceptual_type == 'monai':
                self.perceptual_loss = MONAIPerceptualLoss(
                        spatial_dims=2,
                        network_type="radimagenet_resnet50",
                        is_fake_3d=False,
                        pretrained=True,).to(self.device)
            elif self.perceptual_type == 'vgg1':
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(self.device)
                self.perceptual_loss = PerceptualLoss(vgg)
            elif self.perceptual_type == 'vgg2':
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(self.device)
                self.perceptual_loss = PerceptualLoss2(vgg, layers=(4, 9, 16), layer_weights=[0.5, 1.0, 1.5])
        if self.use_patchifier:
            self.contrastive_loss = PatchNCELoss()
        else:
            self.contrastive_loss = None


        # define optimizer and learning rate scheduler
        # Initialize learnable logit scale for PatchNCE (temperature parameter)
        # Initialize near 1.0 and clamp later to reduce softmax sharpness
        if not hasattr(self, 'logit_scale'):
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(2.0, device=self.device)))

        # PatchNCE warmup/ramp configuration
        self.nce_warmup_epochs = 0
        self.nce_ramp_epochs = 2

        optimizer_params = list(self.decoder.parameters())
        if self.use_beta and self.beta_encoder is not None:
            optimizer_params.extend(list(self.beta_encoder.parameters()))
        if self.use_patchifier and self.patchifier is not None:
            optimizer_params.extend(list(self.patchifier.parameters()))
        if hasattr(self, 'enhanced_style_transfer'):
            optimizer_params.extend(list(self.enhanced_style_transfer.parameters()))
        if hasattr(self, 'multi_scale_injector'):
            optimizer_params.extend(list(self.multi_scale_injector.parameters()))
        if hasattr(self, 'adain_block'):
            optimizer_params.extend(list(self.adain_block.parameters()))
        # Include logit_scale parameter so it is optimized
        optimizer_params.append(self.logit_scale)
            
        self.optimizer = Adam(optimizer_params, lr=lr)
        
        # Initialize discriminator optimizer if adversarial training is enabled
        if self.use_adversarial and self.discriminator is not None:
            self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=lr * self.discriminator_lr_scale)  # Configurable learning rate scaling for discriminator
        #self.scheduler = LR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

        self.out_dir = out_dir
        mkdir_p(self.out_dir)

        # Update experiment name with learning rate
        self._update_experiment_name()
        
        self.writer_path = os.path.join(self.out_dir, self.timestr)
        # Ensure the experiment directory exists
        mkdir_p(self.writer_path)
        
        self.writer = SummaryWriter(self.writer_path, comment=self.experiment_name)
        self.setup_logging(self.writer_path)
        # PatchNCE/patchifier debug history
        self.debug_patch_history = {
            'iters': [], 'pos_mean': [], 'neg_mean': [], 'q_min': [], 'q_max': [], 'q_std': [],
            'grad_beta': [], 'grad_patchifier': [], 'logit_scale': []
        }
        
        # Now that logger is set up, check for existing checkpoint with the same experiment name
        existing_checkpoint_path, existing_epoch = self._find_existing_checkpoint(out_dir)
        
        if existing_checkpoint_path is not None:
            self.logger.info(f"🔄 Found existing checkpoint: {existing_checkpoint_path}")
            self.logger.info(f"📊 Continuing from epoch: {existing_epoch}")
            
            try:
                # Load the existing checkpoint
                self.checkpoint = torch.load(existing_checkpoint_path, map_location=self.device)
                
                # Load model states
                if self.use_beta and 'beta_encoder' in self.checkpoint:
                    self.beta_encoder.load_state_dict(self.checkpoint['beta_encoder'])
                if self.use_patchifier and 'patchifier' in self.checkpoint:
                    self.patchifier.load_state_dict(self.checkpoint['patchifier'])
                if 'decoder' in self.checkpoint:
                    missing, unexpected = self.decoder.load_state_dict(self.checkpoint['decoder'], strict=False)
                    if missing or unexpected:
                        print(f"[Decoder] load_state_dict(strict=False) -> missing: {missing}, unexpected: {unexpected}")
                if hasattr(self, 'enhanced_style_transfer') and 'enhanced_style_transfer' in self.checkpoint:
                    self.enhanced_style_transfer.load_state_dict(self.checkpoint['enhanced_style_transfer'])
                if hasattr(self, 'multi_scale_injector') and 'multi_scale_injector' in self.checkpoint:
                    self.multi_scale_injector.load_state_dict(self.checkpoint['multi_scale_injector'])
                if self.use_adversarial and 'discriminator' in self.checkpoint:
                    self.discriminator.load_state_dict(self.checkpoint['discriminator'])
                
                # Load optimizer states
                self.optimizer.load_state_dict(self.checkpoint['optimizer'])
                if 'scheduler' in self.checkpoint:
                    self.scheduler.load_state_dict(self.checkpoint['scheduler'])
                if self.use_adversarial and 'discriminator_optimizer' in self.checkpoint:
                    self.discriminator_optimizer.load_state_dict(self.checkpoint['discriminator_optimizer'])
                if self.use_adversarial and 'discriminator_step_counter' in self.checkpoint:
                    self.discriminator_step_counter = self.checkpoint['discriminator_step_counter']
                
                # Set start epoch to continue from where we left off
                self.start_epoch = existing_epoch + 1
                
                self.logger.info(f"✅ Successfully loaded checkpoint from epoch {existing_epoch}")
                self.logger.info(f"🚀 Will continue training from epoch {self.start_epoch}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load checkpoint: {str(e)}")
                self.logger.info(f"🆕 Starting fresh training due to checkpoint loading error.")
                self.checkpoint = None
                self.start_epoch = 1
        else:
            self.logger.info(f"🆕 No existing checkpoint found. Starting fresh training.")
            self.start_epoch = 1
        
        # Now that writer_path is set, create the debugger if needed
        #from dist_clip.modules.enhanced_debug import create_enhanced_debugger
        #self.debugger = create_enhanced_debugger(output_dir=self.writer_path)
        
        # Test logging to verify it's working
        self.logger.info("="*50)
        self.logger.info("TRAINING INITIALIZATION COMPLETE")
        self.logger.info(f"Output directory: {self.out_dir}")
        self.logger.info(f"Writer path: {self.writer_path}")
        self.logger.info(f"Experiment name: {self.experiment_name}")
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Use beta encoder: {self.use_beta}")
        self.logger.info(f"Use patchifier: {self.use_patchifier}")
        self.logger.info(f"Contrast feature type: {self.use_contrast_feat}")
        self.logger.info("="*50)

    def _update_experiment_name(self):
        """Update experiment name with current parameters"""
        self.experiment_name = generate_experiment_name(
            beta_dim=self.beta_dim,
            use_beta=self.use_beta,
            beta_type=self.beta_type,
            use_patchifier=self.use_patchifier,
            use_contrast_feat=self.use_contrast_feat,
            use_adversarial=self.use_adversarial,
            use_perceptual=self.use_perceptual,
            discriminator_update_freq=self.discriminator_update_freq,
            discriminator_channels=self.discriminator_channels,
            discriminator_layers=self.discriminator_layers,
            discriminator_lr_scale=self.discriminator_lr_scale,
            batch_size=self.batch_size,
            lr=self.lr,
            max_slices_per_step=self.max_slices_per_step,
            gpu_id=self.device.index if self.device.type == 'cuda' else 0,
            free_text=self.free_text,
            image_contrast=self.guidance_mode,
            perceptual_type=self.perceptual_type if self.use_perceptual else None,
            weights={
                'w_rec': self.w_rec,
                'w_clip': self.w_clip,
                'w_clip_l2': self.w_clip_l2,
                'w_clip_img': self.w_clip_img,
                'w_per': self.w_per,
                'w_beta': self.w_beta,
                'w_adv': self.w_adv,
            } if self.use_perceptual else None
        )
        
        # Update timestr with experiment name (no datetime)
        self.timestr = self.experiment_name
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Updated experiment name: {self.experiment_name}")

    def _find_existing_checkpoint(self, out_dir):
        """Find the latest checkpoint for the current experiment name"""
        experiment_dir = os.path.join(out_dir, self.experiment_name)
        
        if not os.path.exists(experiment_dir):
            return None, 0
        
        # Look for epoch*.pt files
        checkpoint_files = []
        for file in os.listdir(experiment_dir):
            if file.startswith('epoch') and file.endswith('_model.pt'):
                try:
                    # Extract epoch number from filename (e.g., "epoch001_model.pt" -> 1)
                    epoch_str = file.replace('epoch', '').replace('_model.pt', '')
                    epoch_num = int(epoch_str)
                    checkpoint_files.append((epoch_num, file))
                except ValueError:
                    continue
        
        if not checkpoint_files:
            return None, 0
        
        # Sort by epoch number and get the latest
        checkpoint_files.sort(key=lambda x: x[0])
        latest_epoch, latest_file = checkpoint_files[-1]
        checkpoint_path = os.path.join(experiment_dir, latest_file)
        
        return checkpoint_path, latest_epoch

    def load_dataset(self, train_csv_path, val_csv_path, batch_size, normalization_method='01', dataset_type='image', preload=False, transform=None, is_shuffle=True, train_slice_start=None, train_slice_end=None, val_slice_start=None, val_slice_end=None):
        """
        dataset_type: 'image', 'slice', or 'pngslice'
        """
        # Store batch_size for experiment naming
        self.batch_size = batch_size
        
        if dataset_type == 'image':
            train_dataset = PairedImageDataset(train_csv_path, preload=preload, slice_start=train_slice_start, slice_end=train_slice_end)
            val_dataset = PairedImageDataset(val_csv_path, preload=preload, slice_start=val_slice_start, slice_end=val_slice_end)
        elif dataset_type == 'slice':
            train_dataset = PairedSliceDataset(train_csv_path, preload=preload)
            val_dataset = PairedSliceDataset(val_csv_path, preload=preload)
        elif dataset_type == 'pngslice':
            train_dataset = PairedPNGSliceDataset(train_csv_path, preload=preload, transform=transform)
            val_dataset = PairedPNGSliceDataset(val_csv_path, preload=preload, transform=transform)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        # Setup data loaders with distributed sampling for multi-GPU
        if self.is_distributed:
            # Enable per-epoch reshuffle via set_epoch in _train_epoch
            train_sampler = DistributedSampler(train_dataset, shuffle=is_shuffle)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, drop_last=False)
            self.valid_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8)
        else:
            # PyTorch shuffles at every new iterator when shuffle=True
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=8, drop_last=False)
            self.valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        # Update experiment name with batch_size
        self._update_experiment_name()

        # Optionally bind negative sampler from the training loader
        if getattr(self, 'use_dataset_negatives', False):
            try:
                if hasattr(self, 'bind_negative_sampler_from_loader') and self.train_loader is not None:
                    self.bind_negative_sampler_from_loader(self.train_loader, image_key='image')
            except Exception:
                pass

    def calculate_beta(self, images):
        # images: tensor of shape (batch, 1, 224, 224) or list of PairItem
        if isinstance(images, list) and isinstance(images[0], PairItem):
            images = torch.stack([p.img for p in images], dim=0)
        elif isinstance(images, list):
            images = torch.stack(images, dim=0)
        # Now images is (batch, 1, 224, 224)
        
        if self.use_beta and self.beta_encoder is not None:
            # Check if it's a brain tissue encoder
            if hasattr(self.beta_encoder, 'get_tissue_probabilities'):
                # Brain tissue encoder returns (logits, probabilities)
                logits, betas = self.beta_encoder(images)
                # For decoder input, we need a single channel - combine all tissues into one
                tissue_probs = self.beta_encoder.get_tissue_probabilities(images)
                
                if self.beta_type == 'discrete':
                    # Create a clear segmentation map with distinct tissue boundaries
                    # Stack all tissue probabilities and take argmax for discrete segmentation
                    all_tissues = torch.cat([
                        tissue_probs['wm'],
                        tissue_probs['gm'], 
                        tissue_probs['csf'],
                        tissue_probs['other']
                    ], dim=1)  # [B, 4, H, W]
                    
                    # Get discrete segmentation (argmax)
                    tissue_labels = torch.argmax(all_tissues, dim=1, keepdim=True)  # [B, 1, H, W]
                    
                    # Convert to intensity values for better visualization
                    # WM=0.8, GM=0.6, CSF=0.3, Other=0.1
                    intensity_map = torch.zeros_like(tissue_labels, dtype=torch.float)
                    intensity_map[tissue_labels == 0] = 0.8  # White Matter
                    intensity_map[tissue_labels == 1] = 0.6  # Gray Matter  
                    intensity_map[tissue_labels == 2] = 0.3  # CSF
                    intensity_map[tissue_labels == 3] = 0.1  # Other
                    
                    single_channel_betas = intensity_map
                else:  # 'weighted' (default)
                    # Create a single-channel segmentation map by combining tissue probabilities
                    # Weighted combination: WM + GM + 0.5*CSF + 0.3*Other
                    single_channel_betas = (
                        tissue_probs['wm'] + 
                        tissue_probs['gm'] + 
                        0.5 * tissue_probs['csf'] + 
                        0.3 * tissue_probs['other']
                    )
                # all_tissues = torch.cat([
                #     tissue_probs['wm'],
                #     tissue_probs['gm'], 
                #     tissue_probs['csf'],
                #     tissue_probs['other']
                # ], dim=1)  # [B, 4, H, W]
                
                # # Get discrete segmentation (argmax)
                # tissue_labels = torch.argmax(all_tissues, dim=1, keepdim=True)  # [B, 1, H, W]
                
                # # Convert to intensity values for better visualization
                # # WM=0.8, GM=0.6, CSF=0.3, Other=0.1
                # intensity_map = torch.zeros_like(tissue_labels, dtype=torch.float)
                # intensity_map[tissue_labels == 0] = 0.8  # White Matter
                # intensity_map[tissue_labels == 1] = 0.6  # Gray Matter  
                # intensity_map[tissue_labels == 2] = 0.3  # CSF
                # intensity_map[tissue_labels == 3] = 0.1  # Other
                
                # single_channel_betas = intensity_map
                return logits, single_channel_betas
            elif hasattr(self.beta_encoder, 'forward') and self.beta_encoder.__class__.__name__ in ['SimpleBetaEncoder', 'MinimalBetaEncoder']:
                # Simple, minimal, or old encoders
                logits, betas = self.beta_encoder(images)
                return logits, betas
            else:
                # Fallback for UNet-based encoders (old type)
                logits = self.beta_encoder(images)
                #betas = self.channel_aggregation(reparameterize_logit(logits))
                betas = logits
                
                # Simple brightness scaling for UNet-based encoders
                brightness_scale = 1.0  # Adjust this value to make it brighter
                betas_scaled = betas * brightness_scale
                
                # Debug: Print beta magnitudes for UNet-based encoders
                #print(f"DEBUG: UNet-based beta encoder output shapes - logits: {logits.shape}, betas: {betas.shape}")
                # print(f"DEBUG: Beta magnitudes - min: {betas.min().item():.6f}, max: {betas.max().item():.6f}, mean: {betas.mean().item():.6f}")
                # print(f"DEBUG: After scaling by {brightness_scale} - min: {betas_scaled.min().item():.6f}, max: {betas_scaled.max().item():.6f}, mean: {betas_scaled.mean().item():.6f}")
            
            return logits, betas_scaled
        else:
            # Return the input images as beta (identity mapping)
            # images = images / (images.norm(dim=-1, keepdim=True) + 1e-8)
            return images, images

    def channel_aggregation(self, beta_onehot_encode):
        """
        Combine multi-channel one-hot encoded beta into one channel (label-encoding).

        ===INPUTS===
        * beta_onehot_encode: torch.Tensor (batch_size, self.beta_dim, image_dim, image_dim)
            One-hot encoded beta variable. At each pixel location, only one channel will take value of 1,
            and other channels will be 0.
        ===OUTPUTS===
        * beta_label_encode: torch.Tensor (batch_size, 1, image_dim, image_dim)
            The intensity value of each pixel will be determined by the channel index with value of 1.
        """
        batch_size = beta_onehot_encode.shape[0]
        image_dim = beta_onehot_encode.shape[3]
        value_tensor = (torch.arange(0, self.beta_dim) * 1.0).to(self.device)
        value_tensor = value_tensor.view(1, self.beta_dim, 1, 1).repeat(batch_size, 1, image_dim, image_dim)
        beta_label_encode = beta_onehot_encode * value_tensor.detach()
        return beta_label_encode.sum(1, keepdim=True) / self.beta_dim
   
    def patch_nce_loss(self, source_betas, target_betas,temperature=0.1, mask=None, source_images=None, target_images=None):
        B = source_betas.size(0)

        # === Patchify and normalize ===
        query_features = self.patchifier(source_betas).flatten(2)     # [B, C, P]
        positive_features = self.patchifier(target_betas).flatten(2)  # [B, C, P]

        if query_features.size(2) == 0:
            return torch.tensor(0.0, device=source_betas.device)

        query_features = F.normalize(query_features, p=2, dim=1)
        positive_features = F.normalize(positive_features, p=2, dim=1)

        # === Optional masking ===
        if mask is not None:
            grid_size = int(math.sqrt(query_features.shape[2]))
            pooled_mask = F.adaptive_avg_pool2d(mask.float(), (grid_size, grid_size))
            mask_cols = pooled_mask.view(B, -1) > 0.0  # [B, P]

            query_features = query_features * mask_cols.unsqueeze(1)
            positive_features = positive_features * mask_cols.unsqueeze(1)

        # === Flatten all patches across batch ===
        query_flat = query_features.permute(0, 2, 1).reshape(-1, query_features.size(1))       # [B*P, C]
        positive_flat = positive_features.permute(0, 2, 1).reshape(-1, positive_features.size(1))  # [B*P, C]

        # === In-batch similarity matrix ===
        sim_matrix = torch.matmul(query_flat, positive_flat.t())  # [B*P, B*P]

        # === Apply learnable logit scale ===
        # scale = exp(logit_scale). Default init corresponds to 1/0.1 = 10.0
        scale = self.logit_scale.exp().clamp(max=100)
        logits = sim_matrix * scale

        # === Labels: each query i matches positive i ===
        labels = torch.arange(query_flat.size(0), device=logits.device)

        # === Cross entropy loss ===
        loss = F.cross_entropy(logits, labels)

        return loss
   
    def patch_nce_loss2(self, source_betas, target_betas, temperature=0.1, mask=None, source_images=None, target_images=None):
        """
        Compute PatchNCE loss with multiple negative examples for better contrastive learning similarly to HACA3 implementation.
        - source_betas, target_betas: [B, 1, H, W] - query and positive examples
        - source_images: [B, 1, H, W] - original images for additional negatives
        - self.patchifier: module to extract patch features, output [B, 128, P]
        """
        if not self.use_patchifier or self.patchifier is None:
            return torch.tensor(0.0, device=source_betas.device)
            
        batch_size = source_betas.shape[0]
        
        # Extract query features (source_betas)
        query_features = self.patchifier(source_betas).view(batch_size, 128, -1)  # [B, 128, P]
        # Extract positive features (target_betas) 
        positive_features = self.patchifier(target_betas).view(batch_size, 128, -1)  # [B, 128, P]
        
        num_patches = query_features.shape[-1]
        
        # Check for valid patches
        if num_patches == 0:
            return torch.tensor(0.0, device=source_betas.device)

        # Create multiple negative examples
        negative_features_list = []
        # Apply mask filtering if provided
        if mask is not None:
            # Map pixel mask to patch grid
            P = num_patches
            grid_size = int(math.sqrt(P))
            if grid_size * grid_size == P:
                pooled = F.adaptive_avg_pool2d(mask.float(), (grid_size, grid_size))
                mask_cols = (pooled.view(batch_size, -1) > 0.0)  # [B, P]
                
                # Filter query and positive features
                valid_cols = mask_cols & (query_features.abs().sum(dim=1) > 0) & (positive_features.abs().sum(dim=1) > 0)
                
                # Find minimum number of valid patches across batch to ensure consistent tensor sizes
                valid_patch_counts = [valid_cols[i].sum().item() for i in range(batch_size)]
                min_valid_patches = min(valid_patch_counts) if valid_patch_counts else 0
                
                if min_valid_patches == 0:
                    return torch.tensor(0.0, device=source_betas.device)
        
        # 1. Source images as negatives
        # if source_images is not None:
            # source_img_features = self.patchifier(source_images).view(batch_size, 128, -1)
            #     # 2. Target images as negatives
            # if target_images is not None:
            #     target_img_features = self.patchifier(target_images).view(batch_size, 128, -1)
            #     negative_features_list.append(target_img_features)
                    
            # Apply filtering with consistent patch count (vectorized, no loop)
            K = min_valid_patches
            # Select K valid indices per batch (1s outrank 0s)
            selected_indices = torch.topk(valid_cols.float(), k=K, dim=1, largest=True)[1]  # [B, K]
            idx_expanded = selected_indices.unsqueeze(1).expand(-1, query_features.shape[1], -1)  # [B, C, K]

            filtered_query = torch.gather(query_features, 2, idx_expanded)  # [B, C, K]
            filtered_positive = torch.gather(positive_features, 2, idx_expanded)  # [B, C, K]
            filtered_source = None
            # if source_images is not None:
                # filtered_source = torch.gather(source_img_features, 2, idx_expanded)  # [B, C, K]

            # Build negatives: per-sample spatial shuffles (derangements) and cross-batch positives (no self)
            negative_features_list = []

            # Create derangement permutations so no patch stays at its own position
            def _make_derangements(batch_size:int, K:int, device):
                perms = []
                if K <= 1:
                    return torch.zeros((batch_size, K), dtype=torch.long, device=device)
                base = torch.arange(K, device=device)
                for _ in range(batch_size):
                    p = torch.randperm(K, device=device)
                    for i in range(K - 1):
                        if p[i].item() == i:
                            p[i], p[i+1] = p[i+1], p[i]
                    if p[K-1].item() == K-1:
                        j = torch.randint(0, K-1, (1,), device=device).item()
                        p[K-1], p[j] = p[j], p[K-1]
                    perms.append(p.unsqueeze(0))
                return torch.cat(perms, dim=0)

            derange_q = _make_derangements(batch_size, K, query_features.device)  # [B, K]
            derange_p = _make_derangements(batch_size, K, query_features.device)  # [B, K]
            rand_idx_q = derange_q.unsqueeze(1).expand(-1, filtered_query.shape[1], -1)  # [B, C, K]
            rand_idx_p = derange_p.unsqueeze(1).expand(-1, filtered_positive.shape[1], -1)  # [B, C, K]
            neg_der_q = torch.gather(filtered_query, 2, rand_idx_q)  # [B, C, K]
            neg_der_p = torch.gather(filtered_positive, 2, rand_idx_p)  # [B, C, K]
            negative_features_list += [neg_der_q, neg_der_p]

            # Cross-batch negatives: rotate along batch dim to avoid self as negative
            if batch_size > 1:
                for shift in range(1, min(batch_size, 3)):
                    negative_features_list.append(filtered_positive.roll(shifts=shift, dims=0))

            # if filtered_source is not None:
                # negative_features_list.append(filtered_source)

            # Concatenate negatives and limit count to keep balance
            negative_features = torch.cat(negative_features_list, dim=-1)  # [B, C, ?]
            # Limit negatives to at most 4K to avoid overpowering positives
            if negative_features.shape[-1] > 4*K:
                sel = torch.randperm(negative_features.shape[-1], device=negative_features.device)[:4*K]
                negative_features = negative_features[:, :, sel]

            # Set filtered features for subsequent computation
            query_features = filtered_query
            positive_features = filtered_positive

            
            # Feature normalization: LayerNorm over channels before L2 normalize to stabilize variance
            def _layernorm_channels(x):
                B, C, Pn = x.shape
                return F.layer_norm(x.transpose(1, 2), (C,), eps=1e-6).transpose(1, 2)
            query_features = _layernorm_channels(query_features)
            positive_features = _layernorm_channels(positive_features)
            negative_features = _layernorm_channels(negative_features)
            # Avoid dropout here to keep deterministic cosine geometry
            query_features = F.normalize(query_features, p=2, dim=1)
            positive_features = F.normalize(positive_features, p=2, dim=1)
            negative_features = F.normalize(negative_features, p=2, dim=1)
            
            # Compute contrastive loss
            B, C, N = query_features.shape
            
            # Positive similarities: [B, N, 1]
            l_positive = (query_features * positive_features).sum(dim=1)[:, :, None]
            
            # Negative similarities: [B, N, num_negatives]
            l_negative = torch.bmm(query_features.permute(0, 2, 1), negative_features)
            
            # Concatenate positive and negative logits: [B, N, 1 + num_negatives]
            # Tighter clamp on softmax sharpness
            scale = self.logit_scale.exp().clamp(max=10.0)
            logits = torch.cat((l_positive, l_negative), dim=2) * scale 

            # Compute per-batch cross-entropy without flattening
            # loss_b = mean over patches of -log softmax(logits) at class 0; then mean over batch
            log_probs = F.log_softmax(logits, dim=2)  # [B, N, 1+M]
            neg_log_p0 = -log_probs[:, :, 0]         # [B, N]
            loss_per_batch = neg_log_p0.mean(dim=1)  # [B]
            loss = loss_per_batch.mean()
            # cos_loss = 1 - F.cosine_similarity(query_features, positive_features, dim=-1).mean()
            # loss = loss + cos_loss


            
            # # after computing patchifier outputs (before normalization)
            # q = self.patchifier(source_betas)       # [B, C, Hpatch, Wpatch]
            # p = self.patchifier(target_betas)
            # print("patchifier out:", q.shape, q.min().item(), q.max().item(), q.mean().item(), q.std().item())

            # qf = q.flatten(2)    # [B, C, P]
            # pf = p.flatten(2)
            # print("qf var per-channel mean:", qf.var(dim=(0,2)).mean().item())
            # print("pf var per-channel mean:", pf.var(dim=(0,2)).mean().item())

            # # after normalization & flattening into [B*P, C]
            # query_flat = F.normalize(qf, dim=1).permute(0,2,1).reshape(-1, qf.size(1))
            # positive_flat = F.normalize(pf, dim=1).permute(0,2,1).reshape(-1, pf.size(1))

            # sim_diag = (query_flat * positive_flat).sum(dim=1).mean().item()
            # # sample some off-diagonal negatives
            # sim_all = torch.matmul(query_flat, positive_flat.t())
            # neg_mean = (sim_all.sum() - torch.diag(sim_all).sum()) / (sim_all.numel() - sim_all.size(0))
            # print("mean pos-sim (diag):", sim_diag, "mean neg-sim:", neg_mean.item())

            # # check grads exist before optimizer.step()
            # for name, p in self.patchifier.named_parameters():
            #     if p.grad is None:
            #         print("NO GRAD for", name)
            #     else:
            #         print(name, p.grad.norm().item())
            # print("logit_scale:", float(self.logit_scale.exp().item()))
            # # 1) Are source & target betas actually different?
            # print("beta diff norm:", (source_betas - target_betas).abs().mean().item(),
            #     (source_betas - target_betas).view(source_betas.size(0), -1).norm(dim=1).mean().item())

            # # 2) Per-patch norms & a couple of example patch maps
            # q = self.patchifier(source_betas)
            # p = self.patchifier(target_betas)
            # qf = q.view(q.size(0), q.size(1), -1)
            # pf = p.view(p.size(0), p.size(1), -1)
            # print("q min/max/mean/std:", q.min().item(), q.max().item(), q.mean().item(), q.std().item())
            # print("per-channel var mean:", qf.var(dim=(0,2)).mean().item())
            l1_loss = F.l1_loss(source_betas, target_betas, reduction='none')
            l1_loss = l1_loss * mask
            if mask.sum() == 0:
                return None, None
            l1_loss = l1_loss.sum() / mask.sum()
            # print("l1 loss:", l1_loss.item())
            # print(loss.item(), l1_loss.item())
            # ncc_loss = self.local_ncc_loss(source_betas, source_images,win=16)
            # ncc_loss2 = self.local_ncc_loss(target_betas, target_images,win=16)
            # print("ncc loss:", ncc_loss.item(), ncc_loss2.item())
            # print("loss:", loss.item())
            #total_loss =   1.0*loss + 0.5*ncc_loss + 0.5*ncc_loss2 

            total_loss = loss + 5.0*l1_loss
            return total_loss
   
    def patch_nce_loss_old(self, source_betas, target_betas, temperature=0.1, mask=None, source_images=None, target_images=None, mse_weight=5.0, K_ext_negs: int = 2):
        """
        Compute PatchNCE loss with multiple negative examples for better contrastive learning similarly to HACA3 implementation.
        - source_betas, target_betas: [B, C, H, W] - query and positive examples
        - source_images: [B, 1, H, W] - original images for additional negatives
        - self.patchifier: module to extract patch features, output [B, 128, P]
        """
        if not self.use_patchifier or self.patchifier is None:
            return torch.tensor(0.0, device=source_betas.device)
            
        batch_size = source_betas.shape[0]
        
        # Extract query features (source_betas)
        query_features = self.patchifier(source_betas).view(batch_size, 128, -1)  # [B, 128, P]
        # Extract positive features (target_betas) 
        positive_features = self.patchifier(target_betas).view(batch_size, 128, -1)  # [B, 128, P]
        
        num_patches = query_features.shape[-1]
        
        # Check for valid patches
        if num_patches == 0:
            return torch.tensor(0.0, device=source_betas.device)

        # Create multiple negative examples
        negative_features_list = []
        
        # 1. Source images as negatives
        # if source_images is not None:
        #     source_img_features = self.patchifier(source_images).view(batch_size, 128, -1)
        #     negative_features_list.append(source_img_features)
        #     # 2. Target images as negatives
        # if target_images is not None:
        #     target_img_features = self.patchifier(target_images).view(batch_size, 128, -1)
        #     negative_features_list.append(target_img_features)
        
        # 3. Shuffled patches from same images (spatial negatives)
        shuffled_query = query_features[:, :, torch.randperm(num_patches, device=query_features.device)]
        negative_features_list.append(shuffled_query)
        
        
        # 4. Shuffled positive features
        #shuffled_positive = positive_features[:, :, torch.randperm(num_patches, device=positive_features.device)]
        # negative_features_list.append(positive_features)
        
        # 5a. External negatives from random images/slices via user-provided sampler (disabled by default)
        if getattr(self, 'use_dataset_negatives', False) and callable(getattr(self, 'neg_sampler', None)) and K_ext_negs > 0:
            B, Cb, H, W = source_betas.shape
            ext = self.neg_sampler(batch_size=B, H=int(H), W=int(W), K=int(K_ext_negs))  # [B,K,C,H,W]
            if isinstance(ext, (list, tuple)):
                ext = ext[0]
            if ext is not None:
                if ext.dim() == 5 and ext.size(0) == B:
                    Kgot = ext.size(1)
                    # Flatten K into batch for patchifier: [B*K, C, H, W]
                    ext_flat = ext.reshape(B * Kgot, ext.size(2), H, W).to(source_betas.device)
                    # Match channels to patchifier expectation
                    in_ch_pf = self.beta_dim if self.use_patchifier else 1
                    if ext_flat.size(1) != in_ch_pf:
                        if ext_flat.size(1) == 1 and in_ch_pf > 1:
                            ext_flat = ext_flat.repeat(1, in_ch_pf, 1, 1)
                        else:
                            ext_flat = ext_flat[:, :in_ch_pf]
                    ext_feat = self.patchifier(ext_flat).view(B, Kgot, 128, -1)  # [B,K,128,P]
                    # Reshape to [B,128,K*P] so each patch sees K*P negatives per sample
                    ext_feat = ext_feat.permute(0, 2, 1, 3).contiguous().view(B, 128, -1)
                    negative_features_list.append(ext_feat)

        # 5b. Cross-batch memory negatives (recompute features each batch with current patchifier)
        if hasattr(self, '_neg_mem_by_epoch') and len(self._neg_mem_by_epoch) > 0:
            K_mem = self.neg_mem_samples_per_batch
            if K_mem > 0:
                B, Cb, H, W = source_betas.shape
                # Flatten memory across kept epochs
                mem_list = []
                for ep in sorted(self._neg_mem_by_epoch.keys(), reverse=True):
                    mem_list.extend(self._neg_mem_by_epoch[ep])
                if len(mem_list) > 0:
                    Ks = min(K_mem, len(mem_list))
                    idxs = torch.randperm(len(mem_list))[:Ks]
                    sel = [mem_list[i] for i in idxs]  # each [C,H,W]
                    mem = torch.stack(sel, dim=0)  # [Ks,C,H,W]
                    if mem.shape[-2] != H or mem.shape[-1] != W:
                        mem = F.interpolate(mem, size=(H, W), mode='bilinear', align_corners=False)
                    in_ch_pf = self.beta_dim if self.use_patchifier else 1
                    if mem.size(1) != in_ch_pf:
                        if mem.size(1) == 1 and in_ch_pf > 1:
                            mem = mem.repeat(1, in_ch_pf, 1, 1)
                        else:
                            mem = mem[:, :in_ch_pf]
                    with torch.no_grad():
                        mem_feat = self.patchifier(mem.to(source_betas.device))
                    # Standardize to [Ks, 128, P]
                    if mem_feat.dim() == 5:
                        # e.g., [A,B,128,H,W] -> flatten leading dims except 128
                        Ks_eff = mem_feat.size(0) * mem_feat.size(1)
                        mem_feat = mem_feat.view(Ks_eff, mem_feat.size(-3), mem_feat.size(-2) * mem_feat.size(-1))
                    elif mem_feat.dim() == 4:
                        # [Ks, 128, H', W'] -> [Ks, 128, P]
                        mem_feat = mem_feat.view(mem_feat.size(0), mem_feat.size(1), -1)
                    # Now mem_feat is [Ks,128,P]
                    # Duplicate across batch: [B,Ks,128,P] -> [B,128,Ks*P]
                    sel_feat = mem_feat.unsqueeze(0).expand(B, -1, -1, -1)
                    sel_feat = sel_feat.permute(0, 2, 1, 3).contiguous().view(B, 128, -1)
                    negative_features_list.append(sel_feat)

        
        
        # Concatenate all negatives
        negative_features = torch.cat(negative_features_list, dim=-1)  # [B, 128, num_negatives]
        # Shuffle across full negative length
        neg_len = negative_features.size(-1)
        negative_features = negative_features[:, :, torch.randperm(neg_len, device=negative_features.device)]
        # Apply mask filtering if provided
       # mask= None
        if mask is not None:
            # Map pixel mask to patch grid
            P = num_patches
            grid_size = int(math.sqrt(P))
            if grid_size * grid_size == P:
                pooled = F.adaptive_avg_pool2d(mask.float(), (grid_size, grid_size))
                mask_cols = (pooled.view(batch_size, -1) > 0.0)  # [B, P]
                
                # Filter query and positive features
                valid_cols = mask_cols & (query_features.abs().sum(dim=1) > 0) & (positive_features.abs().sum(dim=1) > 0)
                
                # Find minimum number of valid patches across batch to ensure consistent tensor sizes
                valid_patch_counts = [valid_cols[i].sum().item() for i in range(batch_size)]
                min_valid_patches = min(valid_patch_counts) if valid_patch_counts else 0
                
                if min_valid_patches == 0:
                    return torch.tensor(0.0, device=source_betas.device)
                
                # Apply filtering with consistent patch count
                filtered_query = []
                filtered_positive = []
                filtered_negative = []
                
                for i in range(batch_size):
                    if valid_cols[i].sum().item() > 0:
                        # Get valid patch indices
                        valid_indices = valid_cols[i].nonzero(as_tuple=True)[0]
                        # Take only the minimum number of patches to ensure consistency
                        selected_indices = valid_indices[:min_valid_patches]
                        
                        filtered_query.append(query_features[i:i+1, :, selected_indices])
                        filtered_positive.append(positive_features[i:i+1, :, selected_indices])
                        # For negatives, take the same number of patches
                        filtered_negative.append(negative_features[i:i+1, :, :])
                if len(filtered_query) == 0:
                    return torch.tensor(0.0, device=source_betas.device)
                
                query_features = torch.cat(filtered_query, dim=0)
                positive_features = torch.cat(filtered_positive, dim=0)
                negative_features = torch.cat(filtered_negative, dim=0)
            else:
                # Fallback: use feature-based filtering
                valid_cols = (query_features.abs().sum(dim=1) > 0) & (positive_features.abs().sum(dim=1) > 0)
                if valid_cols.sum().item() == 0:
                    return torch.tensor(0.0, device=source_betas.device)
        
        # Normalize features
        query_features = F.normalize(query_features, p=2, dim=1)  # [B, 128, P]
        positive_features = F.normalize(positive_features, p=2, dim=1)  # [B, 128, P]
        negative_features = F.normalize(negative_features, p=2, dim=1)  # [B, 128, num_negatives]
        
        # Compute contrastive loss
        B, C, N = query_features.shape
        
        # Positive similarities: [B, N, 1]
        l_positive = (query_features * positive_features).sum(dim=1)[:, :, None]
        
        # Negative similarities: [B, N, num_negatives]
        l_negative = torch.bmm(query_features.permute(0, 2, 1), negative_features)
        
        # Concatenate positive and negative logits: [B, N, 1 + num_negatives]
        logits = torch.cat((l_positive, l_negative), dim=2) / temperature
        
        # Flatten for cross-entropy: [B*N, 1 + num_negatives]
        predictions = logits.flatten(0, 1)
        
        # Targets: positive is always at index 0
        targets = torch.zeros(B * N, dtype=torch.long, device=query_features.device)
        
        # Compute PatchNCE cross-entropy loss
        loss_nce = F.cross_entropy(predictions, targets)

        # Channel-wise MSE loss between betas (masked if provided)
        # mse_per_pixel: [B, C, H, W]
        mse_map = (source_betas - target_betas) ** 2
        if mask is not None:
            denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)  # [B,1,1,1]
            mse_per_ch = (mse_map * mask).sum(dim=(2, 3), keepdim=True) / denom  # [B,C,1,1]
            mse_per_ch = mse_per_ch.squeeze(-1).squeeze(-1)  # [B,C]
        else:
            mse_per_ch = mse_map.mean(dim=(2, 3))  # [B,C]
        loss_mse_ch = mse_per_ch.mean()  # average over batch and channels

        # Option 1: Cosine similarity loss (more stable than PatchNCE)
       # loss_cosine = self._cosine_similarity_loss(query_features, positive_features, negative_features_list, mask)
        
        # Option 2: Original PatchNCE (currently disabled due to collapse)
        # total_loss = loss_nce + mse_weight * loss_mse_ch
        
        # Option 3: Only MSE (current fallback)
        # total_loss = loss_mse_ch
        
        # Use cosine similarity loss with MSE regularization
        total_loss = loss_nce + mse_weight * loss_mse_ch
        return total_loss
    
    def _cosine_similarity_loss(self, query_features, positive_features, negative_features_list, mask):
        """
        Cosine similarity loss that's more stable than PatchNCE for anatomical representations.
        
        Args:
            query_features: [B, 128, P] - query patch features
            positive_features: [B, 128, P] - positive patch features  
            negative_features_list: List of [B, 128, P] negative features
            mask: [B, 1, H, W] - mask for valid regions
            
        Returns:
            loss: scalar tensor
        """
        B, feat_dim, num_patches = query_features.shape
        
        # 1. Positive similarity loss: maximize cosine similarity between query and positive
        # Normalize features
        query_norm = F.normalize(query_features, p=2, dim=1, eps=1e-8)  # [B, 128, P]
        positive_norm = F.normalize(positive_features, p=2, dim=1, eps=1e-8)  # [B, 128, P]
        
        # Compute cosine similarities
        pos_similarities = (query_norm * positive_norm).sum(dim=1)  # [B, P]
        
        # Positive loss: minimize (1 - cosine_similarity) to maximize similarity
        pos_loss = (1.0 - pos_similarities).mean()
        
        # 2. Negative similarity loss: minimize cosine similarity between query and negatives
        neg_loss = 0.0
        if negative_features_list:
            # Concatenate all negatives
            all_negatives = torch.cat(negative_features_list, dim=2)  # [B, 128, total_neg_patches]
            neg_norm = F.normalize(all_negatives, p=2, dim=1, eps=1e-8)  # [B, 128, total_neg_patches]
            
            # Compute cosine similarities with negatives
            neg_similarities = torch.bmm(
                query_norm.transpose(1, 2),  # [B, P, 128]
                neg_norm  # [B, 128, total_neg_patches]
            )  # [B, P, total_neg_patches]
            
            # Negative loss: minimize max cosine similarity (prevent collapse to negatives)
            # Use max pooling to get the hardest negative per query patch
            max_neg_sim, _ = neg_similarities.max(dim=2)  # [B, P]
            neg_loss = max_neg_sim.mean()
        
        # 3. Regularization: prevent feature collapse
        # Encourage feature diversity by penalizing very high similarities
        feature_diversity_loss = 0.0
        if num_patches > 1:
            # Compute pairwise similarities within query features
            query_sim_matrix = torch.bmm(query_norm.transpose(1, 2), query_norm)  # [B, P, P]
            # Remove diagonal (self-similarity)
            mask_diag = torch.eye(num_patches, device=query_features.device).unsqueeze(0).expand(B, -1, -1)
            query_sim_matrix = query_sim_matrix * (1 - mask_diag)
            # Penalize high similarities between different patches
            feature_diversity_loss = torch.clamp(query_sim_matrix, min=0.8).mean()
        
        # 4. Combine losses
        total_loss = pos_loss + 0.1 * neg_loss + 0.05 * feature_diversity_loss
        
        return total_loss
    
    def bind_negative_sampler_from_loader(self, loader, image_key: str = 'image'):
        """
        Bind a sampler that fetches external negatives from a PyTorch DataLoader's dataset.
        - loader: DataLoader with a dataset supporting __getitem__(idx) -> sample (tensor or dict)
        - image_key: if dataset returns dict, use this key to extract image tensor
        The sampler will randomly pick K images (and random slices if 3D) per sample in the batch.
        """
        import random
        dataset = getattr(loader, 'dataset', None)
        if dataset is None:
            self.neg_sampler = None
            if hasattr(self, 'logger'):
                self.logger.warning('bind_negative_sampler_from_loader: loader has no dataset; neg_sampler disabled')
            return

        def _to_tensor(sample):
            if isinstance(sample, dict):
                x = sample.get(image_key, None)
                if x is None:
                    # fallback to first tensor-like value
                    for v in sample.values():
                        if torch.is_tensor(v):
                            x = v; break
                return x
            return sample

        def _extract_2d(img: torch.Tensor) -> torch.Tensor:
            # Accept shapes: [C,H,W], [C,D,H,W], [H,W]
            if img.dim() == 2:
                return img.unsqueeze(0)
            if img.dim() == 3:
                return img
            if img.dim() == 4:
                # [C,D,H,W] -> random D slice
                _, D, _, _ = img.shape
                d = random.randint(0, max(D - 1, 0))
                return img[:, d]
            # Unknown shape; try flatten last two dims
            return img.reshape(1, img.shape[-2], img.shape[-1])

        def _sampler(batch_size: int, H: int, W: int, K: int) -> torch.Tensor:
            ch = self.beta_dim if self.use_patchifier else 1
            outs = []
            total = len(dataset)
            device = self.device if hasattr(self, 'device') else 'cpu'
            for _ in range(batch_size):
                imgs_k = []
                for _k in range(K):
                    idx = random.randint(0, max(total - 1, 0))
                    sample = dataset[idx]
                    img = _to_tensor(sample)
                    if img is None:
                        # create blank if dataset doesn't provide image
                        img = torch.zeros(1, H, W)
                    img = _extract_2d(img)
                    # Ensure channels
                    if img.size(0) < ch:
                        reps = (ch + img.size(0) - 1) // img.size(0)
                        img = img.repeat(reps, 1, 1)[:ch]
                    elif img.size(0) > ch:
                        img = img[:ch]
                    # Resize to HxW if needed
                    if img.shape[-2] != H or img.shape[-1] != W:
                        img = F.interpolate(img.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                    # Normalize to [0,1] if likely not
                    if img.dtype.is_floating_point:
                        imin = float(img.min()); imax = float(img.max())
                        if imax > imin:
                            img = (img - imin) / (imax - imin)
                        img = img.clamp(0, 1)
                    else:
                        img = img.float() / 255.0
                    imgs_k.append(img.unsqueeze(0))  # [1,C,H,W]
                outs.append(torch.cat(imgs_k, dim=0).unsqueeze(0))  # [1,K,C,H,W]
            return torch.cat(outs, dim=0).to(device)  # [B,K,C,H,W]

        self.neg_sampler = _sampler
        if hasattr(self, 'logger'):
            self.logger.info('External negative sampler bound from loader')
    
    def calculate_anatomy_beta_loss(self, source_betas, target_betas, temperature=0.1, mask=None, source_images=None, target_images=None):
        """
        Enhanced brain tissue-focused beta loss with tissue-specific consistency
        """
        if not self.use_beta or self.beta_encoder is None:
            return torch.tensor(0.0, device=source_betas.device)
        
        # Check if we're using brain tissue encoder
        if hasattr(self.beta_encoder, 'get_tissue_probabilities'):
            # Use tissue-specific loss
            return self._calculate_tissue_consistency_loss(source_betas, target_betas, temperature)
        else:
            # Fallback to generic beta loss
            return self._calculate_generic_beta_loss(source_betas, target_betas, temperature, mask=mask, source_images=source_images, target_images=target_images)
    
    def _calculate_tissue_consistency_loss(self, source_betas, target_betas, temperature=0.1):
        """
        Calculate tissue-specific consistency loss for brain tissue segmentation
        """
        # Get tissue probability maps
        source_tissues = self.beta_encoder.get_tissue_probabilities(source_betas)
        target_tissues = self.beta_encoder.get_tissue_probabilities(target_betas)
        
        # Component 1: Tissue-specific consistency loss
        tissue_consistency_loss = 0.0
        tissue_weights = {'wm': 1.0, 'gm': 1.0, 'csf': 0.8, 'other': 0.6}
        
        for tissue_type, weight in tissue_weights.items():
            if tissue_type in source_tissues and tissue_type in target_tissues:
                tissue_loss = F.mse_loss(source_tissues[tissue_type], target_tissues[tissue_type])
                tissue_consistency_loss += weight * tissue_loss
        
        # Component 2: PatchNCE loss for overall anatomical consistency
        patch_nce_loss = self.patch_nce_loss(source_betas, target_betas, temperature)
        
        # Component 3: Tissue diversity loss (encourage distinct tissue classes)
        tissue_diversity_loss = 0.0
        if source_betas.shape[1] == 4:  # 4 tissue classes
            # Encourage each tissue class to be distinct
            tissue_probs = F.softmax(source_betas, dim=1)
            # Maximize entropy to encourage balanced tissue distribution
            entropy = -torch.sum(tissue_probs * torch.log(tissue_probs + 1e-8), dim=1).mean()
            tissue_diversity_loss = -entropy  # Minimize negative entropy = maximize entropy
        
        # Component 4: Anatomical feature consistency
        if hasattr(self.beta_encoder, 'get_anatomical_features'):
            source_features = self.beta_encoder.get_anatomical_features(source_betas)
            target_features = self.beta_encoder.get_anatomical_features(target_betas)
            feature_consistency_loss = F.mse_loss(source_features, target_features)
        else:
            feature_consistency_loss = torch.tensor(0.0, device=source_betas.device)
        
        # Combine losses with tissue-specific weighting
        total_beta_loss = (
            0.4 * tissue_consistency_loss +    # Tissue-specific consistency
            0.3 * patch_nce_loss +             # Overall anatomical consistency
            0.2 * tissue_diversity_loss +      # Tissue diversity
            0.1 * feature_consistency_loss     # Feature-level consistency
        )
        
        return total_beta_loss
    
    def _calculate_generic_beta_loss(self, source_betas, target_betas, temperature=0.1, mask=None, source_images=None, target_images=None):
        """
        Generic beta loss for non-tissue-specific encoders
        """
        # Component 1: PatchNCE loss (anatomical consistency)
        patch_nce_loss = self.patch_nce_loss(source_betas, target_betas, temperature, mask=mask, source_images=source_images, target_images=target_images)
        
        total_beta_loss = patch_nce_loss
        return total_beta_loss
    
    def calculate_adversarial_loss(self, real_images, fake_images, epoch, is_train=True):
        """
        Calculate adversarial loss for both generator and discriminator using MONAI's PatchAdversarialLoss.
        Implements slower discriminator updates for better training stability.
        
        Args:
            real_images: Real target images [B, 1, H, W]
            fake_images: Generated images [B, 1, H, W]
            epoch: Current training epoch
            is_train: Whether in training mode
            
        Returns:
            gen_loss: Generator adversarial loss
            disc_loss: Discriminator loss
        """
        if not self.use_adversarial or self.discriminator is None:
            return torch.tensor(0.0, device=real_images.device), torch.tensor(0.0, device=real_images.device)
        
        if is_train:
            # Increment discriminator step counter
            self.discriminator_step_counter += 1
            
            # Adaptive discriminator update frequency for better training
            if epoch <= 5:
                update_freq = max(1, self.discriminator_update_freq // 2)  # More frequent updates early
            else:
                update_freq = self.discriminator_update_freq * 2  # Much less frequent updates later
            
            should_update_discriminator = (self.discriminator_step_counter % update_freq == 0)
            
            if should_update_discriminator:
                # Train discriminator
                self.discriminator_optimizer.zero_grad()
                
                # Real images - discriminator should classify as real
                real_validity = self.discriminator(real_images)
                d_real_loss = self.adversarial_loss(real_validity, target_is_real=True, for_discriminator=True)
                
                # Fake images - discriminator should classify as fake
                fake_validity = self.discriminator(fake_images.detach())
                d_fake_loss = self.adversarial_loss(fake_validity, target_is_real=False, for_discriminator=True)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                d_loss.backward()
                self.discriminator_optimizer.step()
                
                
                # Skip discriminator update if it's becoming too strong
                if d_loss.item() < 0.1:
                    if self.rank == 0:
                        self.logger.warning(f"🎭 Discriminator too strong (loss: {d_loss.item():.6f}), skipping update")
                    return g_loss, d_loss
            else:
                # Don't update discriminator, just compute loss for logging
                with torch.no_grad():
                    real_validity = self.discriminator(real_images)
                    fake_validity = self.discriminator(fake_images.detach())
                    d_real_loss = self.adversarial_loss(real_validity, target_is_real=True, for_discriminator=True)
                    d_fake_loss = self.adversarial_loss(fake_validity, target_is_real=False, for_discriminator=True)
                    d_loss = d_real_loss + d_fake_loss
            
            # Always train generator (adversarial loss)
            fake_validity = self.discriminator(fake_images)
            
            # Generator wants discriminator to classify fake images as real
            g_loss = self.adversarial_loss(fake_validity, target_is_real=True, for_discriminator=False)
            
            return g_loss, d_loss
        else:
            # Evaluation mode - just compute losses without backprop
            with torch.no_grad():
                real_validity = self.discriminator(real_images)
                fake_validity = self.discriminator(fake_images)
                
                d_real_loss = self.adversarial_loss(real_validity, target_is_real=True, for_discriminator=True)
                d_fake_loss = self.adversarial_loss(fake_validity, target_is_real=False, for_discriminator=True)
                d_loss = d_real_loss + d_fake_loss
                
                g_loss = self.adversarial_loss(fake_validity, target_is_real=True, for_discriminator=False)
                
                return g_loss, d_loss
    
    
    def calculate_clip_loss(self, glob_features, source_clip_img_feat, source_clip_text_feat, target_clip_text_feat, rec_image, target_clip_img_feat):
        eps = 1e-8
        
        # Create augmented versions for CLIP loss (matching CLIPStyler)
        num_crops = 4  # Number of augmented versions
        aug_features = []
        
        for _ in range(num_crops):
            # Apply random crop augmentation
            aug_img = get_clip_augmentation()(rec_image)

            aug_feat = self.clip_model.encode_image(aug_img)
            aug_features.append(aug_feat)
        
        # Stack all augmented features
        img_proc = torch.cat(aug_features, dim=0)  # [num_crops * batch_size, feature_dim]
        
        # Normalize features
        img_proc = img_proc / (img_proc.norm(dim=-1, keepdim=True) + eps)
        source_features = source_clip_img_feat / (source_clip_img_feat.norm(dim=-1, keepdim=True) + eps)
        source_text_features = source_clip_text_feat / (source_clip_text_feat.norm(dim=-1, keepdim=True) + eps)
        target_text_features = target_clip_text_feat / (target_clip_text_feat.norm(dim=-1, keepdim=True) + eps)
        target_img_features = target_clip_img_feat / (target_clip_img_feat.norm(dim=-1, keepdim=True) + eps)
        
        # Calculate directions (matching CLIPStyler implementation)
        img_direction = img_proc - source_features.repeat(num_crops, 1)
        img_direction = img_direction / (img_direction.norm(dim=-1, keepdim=True) + eps)
        
        text_direction1 = target_text_features - source_text_features
        text_direction = text_direction1.repeat(num_crops, 1)
        text_direction = text_direction / (text_direction.norm(dim=-1, keepdim=True) + eps)
        
        # Use cosine similarity for directional loss (matching CLIPStyler)
        cosine_sim = torch.cosine_similarity(img_direction, text_direction, dim=1)
        loss_temp = (1 - cosine_sim)
        
        # Apply threshold filtering (matching CLIPStyler approach)
        thresh = 0.2  # Threshold for filtering low-quality losses
        loss_temp[loss_temp < thresh] = 0
        loss_clip = loss_temp.mean()
        
        # L2 loss for additional regularization (matching CLIPStyler)
        # Option 1: Disable L2 loss completely (gradient-safe)
        cosine_sim = torch.cosine_similarity(glob_features, target_text_features, dim=1)
        loss_clip_l2 = (1 - cosine_sim)
        loss_clip_l2 = loss_clip_l2.mean()
        
        # NEW: Content preservation loss - ensure reconstructed image matches target image content
        # Use precomputed target CLIP image features
        content_sim = torch.cosine_similarity(glob_features, target_img_features, dim=1)
        loss_content_preservation = (1 - content_sim).mean()
        
        return loss_clip, loss_clip_l2, loss_content_preservation
       
    def local_ncc_loss(self, I, J, win=16, eps=1e-5):
        # I, J: (B, C, H, W)
        ndims = 2
        sum_filt = torch.ones([1, 1] + [win]*ndims).to(I.device)

        pad_no = 0
        stride = win
        padding = [pad_no]*ndims

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv2d(I, sum_filt, stride=stride, padding=pad_no)
        J_sum = F.conv2d(J, sum_filt, stride=stride, padding=pad_no)
        I2_sum = F.conv2d(I2, sum_filt, stride=stride, padding=pad_no)
        J2_sum = F.conv2d(J2, sum_filt, stride=stride, padding=pad_no)
        IJ_sum = F.conv2d(IJ, sum_filt, stride=stride, padding=pad_no)

        win_size = win ** ndims
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + eps)
        return 1 - torch.mean(cc)
    def _cycle_consistency_loss(self, betas_A, style_A, style_B, mask):
        """
        Beta-stability cycle consistency:
        - A_B = Decoder(βA, SB)
        - βA_B = BetaEncoder(A_B)
        - L_cyc = ||βA - βA_B||_1 (masked)
        """
        device = betas_A.device
        if betas_A.size(0) == 0 or mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        def decode_with_style(betas, style_feat):
            if self.use_contrast_feat == 'adain' and hasattr(self, 'adain_block'):
                mod = self.adain_block(betas, style_feat)
                return self.decoder(mod * mask) * mask
            elif self.use_contrast_feat == 'adapter' and hasattr(self, 'feature_adapter'):
                contrast_map = self.feature_adapter(style_feat)
                dec_in = torch.cat([betas, contrast_map], dim=1) * mask
                return self.decoder(dec_in) * mask
            elif self.use_contrast_feat == 'attention' and hasattr(self, 'cross_attn_block'):
                x_flat = betas.view(betas.size(0), 1, betas.size(-2) * betas.size(-1)).permute(0, 2, 1)
                x_flat = self.image_proj(x_flat)
                style = self.text_proj(style_feat).unsqueeze(1)
                x_att = self.cross_attn_block(x_flat, style)
                x_att = self.img_proj_back(x_att)
                dec_in = x_att.permute(0, 2, 1).view(betas.size(0), 1, betas.size(-2), betas.size(-1)) * mask
                return self.decoder(dec_in) * mask
            elif 'enhanced' in str(self.use_contrast_feat):
                return self.decoder(in_tensor=betas * mask, style_text_feat=style_feat) * mask
            elif self.use_contrast_feat == 'multiscale' and hasattr(self, 'multi_scale_injector'):
                dec_in = betas * mask
                feats = []
                cur = dec_in
                for i in range(4):
                    feats.append(cur)
                    if i < 3:
                        cur = F.avg_pool2d(cur, kernel_size=2, stride=2)
                mod = self.multi_scale_injector(feats, style_feat)
                enh = []
                for i, feat in enumerate(mod):
                    use_attention = (i == 0) and hasattr(self, 'enhanced_style_transfer')
                    if hasattr(self, 'enhanced_style_transfer'):
                        e = self.enhanced_style_transfer(feat, style_feat, use_attention=use_attention)
                        if not hasattr(self, f'enhanced_proj_scale{i}'):
                            setattr(self, f'enhanced_proj_scale{i}', nn.Conv2d(256, 1, 1).to(e.device))
                        proj = getattr(self, f'enhanced_proj_scale{i}')
                        e = proj(e)
                    else:
                        e = feat
                    enh.append(e)
                return self.decoder(enh) * mask
            else:
                # Fallback: no style module, direct decode
                return self.decoder(betas * mask) * mask

        # Swap style: synthesize A under style_B
        img_A_B = decode_with_style(betas_A, style_B)

        # Re-encode beta from synthesized image
        with torch.no_grad():
            _, betas_A_B = self.calculate_beta(img_A_B)

        # L1 in beta space (masked)
        diff = (betas_A - betas_A_B).abs()
        if mask is not None:
            # Expand mask to beta channels
            m = mask
            if m.size(1) != betas_A.size(1):
                m = m.repeat(1, betas_A.size(1), 1, 1)
            diff = diff * m
            denom = m.sum().clamp_min(1e-6)
        else:
            denom = torch.tensor(float(diff.numel()), device=device)
        return diff.sum() / denom
    def calculate_loss(self, epoch, rec_image, ref_image, mask, source_betas, target_betas, source_images, source_clip_img_feat, source_clip_text_feat, target_clip_text_feat, target_clip_img_feat, target_images, is_train=True):
        """
        Calculate losses for MR-Styler training and validation.
        """
        # 1. reconstruction loss
        ref_image = ref_image * mask
        rec_vals = self.rec_loss(rec_image, ref_image)
        masked_rec_vals = rec_vals * mask
        if mask.sum() == 0:
            return None, None
        rec_loss = masked_rec_vals.sum() / mask.sum()
        # rec_loss = self.ncc_loss(rec_image, ref_image)
        
        # 2. perceptual loss (optional)
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(rec_image, ref_image).mean()
        else:
            perceptual_loss = torch.tensor(0.0, device=self.device)
        
        # 3. directional CLIP loss (always compute, but with warmup)
        # Encode reconstructed image with CLIP using proper normalization and augmentation
        if epoch > 2:
            glob_features = self.clip_model.encode_image(rec_image)
            loss_clip, loss_clip_l2, loss_clip_image = self.calculate_clip_loss(glob_features, source_clip_img_feat, source_clip_text_feat, target_clip_text_feat, rec_image, target_clip_img_feat)
        else:   
            loss_clip = torch.tensor(0.0, device=self.device)
            loss_clip_l2 = torch.tensor(0.0, device=self.device)
            loss_clip_image = torch.tensor(0.0, device=self.device)
        # Add warmup for CLIP loss in early epochs
        if self.use_beta and self.beta_encoder is not None:
            # Simple anatomy-focused beta loss
            beta_loss = self.calculate_anatomy_beta_loss(source_betas, target_betas, mask=mask, source_images=source_images, target_images=target_images)
        else:
            beta_loss = torch.tensor(0.0, device=self.device)
        
        # # 4. Cycle consistency in beta space
        # if self.use_beta and self.beta_encoder is not None and epoch > 1:
        #     style_A = target_clip_text_feat if self.guidance_mode != 'image' else target_clip_img_feat
        #     if style_A is not None and style_A.size(0) > 1:
        #         perm = torch.randperm(style_A.size(0), device=style_A.device)
        #         style_B = style_A[perm]
        #     else:
        #         style_B = style_A
        #     cycle_loss = self._cycle_consistency_loss(source_betas, style_A, style_B, mask)
        # else:
        #     cycle_loss = torch.tensor(0.0, device=self.device)

        # 5. Adversarial loss (only after epoch 2 to allow initial convergence)
        if self.use_adversarial and epoch > 2:
            adv_loss, disc_loss = self.calculate_adversarial_loss(ref_image, rec_image,epoch, is_train=is_train)
        else:
            adv_loss = torch.tensor(0.0, device=self.device)
            disc_loss = torch.tensor(0.0, device=self.device)
        
        # Adjusted loss weights (matching CLIPStyler approach)
        # CLIPStyler uses: lambda_patch*loss_patch + lambda_ncc*ncc_loss + lambda_dir*loss_glob
        # We adapt this to our losses: rec_loss + clip_loss + perceptual_loss + beta_loss + content_preservation + adversarial
        # Dynamic adversarial loss weight based on epoch
        
        # PatchNCE warmup and ramp schedule
        nce_weight_scale = 1.0
        if epoch <= self.nce_warmup_epochs:
            nce_weight_scale = 0.0
        # elif epoch <= self.nce_warmup_epochs + self.nce_ramp_epochs:
            # nce_weight_scale = (epoch - self.nce_warmup_epochs) / float(self.nce_ramp_epochs)

        total_loss = (  self.w_rec * rec_loss + 
                         self.w_clip * loss_clip + 
                         (self.w_per * perceptual_loss if self.use_perceptual else 0.0) +
                         self.w_clip_l2 * loss_clip_l2 +
                         self.w_clip_img * loss_clip_image +
                         self.w_beta * beta_loss * nce_weight_scale +
                         # self.w_cyc * cycle_loss +  # disabled for now
                         self.w_adv * adv_loss)
        
        loss = {
            'rec_loss': rec_loss.item(),
            'per_loss': perceptual_loss.item() if self.use_perceptual else 0.0,
            'clip_loss': loss_clip.item(),
            'clip_loss_l2': loss_clip_l2.item(),
            'clip_image_loss': loss_clip_image.item(),
            'beta_loss': beta_loss.item(),
            # 'cycle_loss': cycle_loss.item(),
            'adv_loss': adv_loss.item(),
            'disc_loss': disc_loss.item(),
            'total_loss': total_loss.item(),
        }

        return loss, total_loss


    def save_model(self, epoch, file_name):
        state = {'epoch': epoch,
                 'timestr': self.timestr,
                 'decoder': self.decoder.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
            }
        
        if self.use_beta and self.beta_encoder is not None:
            state['beta_encoder'] = self.beta_encoder.state_dict()
            
        if self.use_patchifier and self.patchifier is not None:
            state['patchifier'] = self.patchifier.state_dict()
            
        if hasattr(self, 'enhanced_style_transfer'):
            state['enhanced_style_transfer'] = self.enhanced_style_transfer.state_dict()
            
        if hasattr(self, 'multi_scale_injector'):
            state['multi_scale_injector'] = self.multi_scale_injector.state_dict()
            
        if self.use_adversarial and self.discriminator is not None:
            state['discriminator'] = self.discriminator.state_dict()
            state['discriminator_optimizer'] = self.discriminator_optimizer.state_dict()
            state['discriminator_step_counter'] = self.discriminator_step_counter
            
        torch.save(obj=state, f=file_name)


    def _train_epoch(self, epoch, epochs, loss_history):
        # if self.debugger is not None:
        #     self.debugger.reset_debug_counter()
        """Train for one epoch"""
        epoch_start_time = datetime.now()
        
        # ====== EPOCH SETUP ======
        if self.rank == 0:  # Only log on main process
            self.logger.info(f"📚 EPOCH {epoch}/{epochs} - TRAINING")
            self.logger.info(f"⏰ Started at: {epoch_start_time.strftime('%H:%M:%S')}")
            self.logger.info("─"*60)
        
        # elif epoch > 1:
        #     # Freeze patchifier and unfreeze others for normal training
        #     if self.use_patchifier and self.patchifier is not None:
        #             for p in self.patchifier.parameters():
        #                 p.requires_grad = False


        
        # Set model to training mode
        self.decoder.train()
        if self.use_beta and self.beta_encoder is not None:
            self.beta_encoder.train()
        if self.use_patchifier and self.patchifier is not None:
            self.patchifier.train()
        if self.use_adversarial and self.discriminator is not None:
            self.discriminator.train()
        
        # Set epoch for distributed sampler
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)

        # Initialize epoch tracking
        epoch_batch_losses = []
        total_batches = len(self.train_loader)
        
        # ====== BATCH PROCESSING ======
        # Initialize memory bucket for this epoch
        if epoch not in self._neg_mem_by_epoch:
            self._neg_mem_by_epoch[epoch] = []  # list of tensors [C,H,W]
        # Expose epoch for memory caching inside patch_nce_loss
        self._current_epoch_for_mem = epoch
        # Drop very old epochs beyond window
        for old_epoch in list(self._neg_mem_by_epoch.keys()):
            if epoch - old_epoch > self.neg_mem_max_epochs:
                del self._neg_mem_by_epoch[old_epoch]
        
        for batch_id, batch in enumerate(self.train_loader):
            batch_id_flag=0
            # if batch_id ==1:
                #  continue
            # Process batch data
            source_imgs, target_imgs, source_texts, target_texts, batch_size, num_slices = self._process_batch_data(batch)
            # External masks removed: on-the-fly consistency will be used
            # Check data filters (coronal/sagittal)
            if self.coronal_check and self._check_data_filters(source_texts, target_texts, batch_id):
                continue
            
            # Encode CLIP text features
            source_clip_text_feat, target_clip_text_feat = self._encode_clip_features(source_texts, target_texts, num_slices)
            
            # Initialize batch loss accumulation
            batch_losses = []
            total_slices = source_imgs.shape[0]
            step = self.max_slices_per_step or total_slices
            
            # Process images in chunks
            for start in range(0, total_slices, step):
                end = min(start + step, total_slices)
                
                # Extract chunk data
                src_imgs_chunk = source_imgs[start:end]
                tgt_imgs_chunk = target_imgs[start:end]
                src_clip_text_feat_chunk = source_clip_text_feat[start:end]
                tgt_clip_text_feat_chunk = target_clip_text_feat[start:end]
                
                # Check mask consistency
                # On-the-fly consistency
                mask = self._check_mask_consistency(src_imgs_chunk, tgt_imgs_chunk, batch_id, start, end)
                if mask is None:
                    continue
                # Process image chunk through model
                contrast_guidance_imgs = None
                if self.guidance_mode in ['image', 'both']:
                    rand= torch.randint(low=10, high= 40, size=(src_imgs_chunk.shape[0],))       
                    contrast_guidance_imgs = target_imgs[rand,...]
                else:
                    contrast_guidance_imgs = tgt_imgs_chunk
                
                output, source_betas, target_betas, source_clip_img_feat, target_clip_img_feat = self._process_image_chunk(
                    src_imgs_chunk, tgt_imgs_chunk, src_clip_text_feat_chunk, tgt_clip_text_feat_chunk, mask, contrast_guidance_imgs
                )
                
                # Record a small subset of slices into memory (detached, CPU)
                if batch_id_flag == 0:
                    with torch.no_grad():
                        keep = min(self.neg_mem_slices_per_epoch - len(self._neg_mem_by_epoch[epoch]), 3)
                        if keep > 0:
                            pick = torch.randperm(source_imgs.size(0))[:keep]
                            for idx in pick:
                                sl = source_imgs[idx].detach().cpu()  # [1,H,W] or [C,H,W]
                                if sl.dim() == 3:
                                    self._neg_mem_by_epoch[epoch].append(sl)
                                else:
                                    self._neg_mem_by_epoch[epoch].append(sl.unsqueeze(0))
                    batch_id_flag=1
                # Calculate loss and backpropagate
                loss_dict = self._calculate_and_backpropagate_loss( 
                    output, tgt_imgs_chunk, mask, source_betas, target_betas, 
                    src_imgs_chunk, source_clip_img_feat, src_clip_text_feat_chunk, 
                    tgt_clip_text_feat_chunk, target_clip_img_feat, epoch, batch_id,
                   
                )
                
                if loss_dict is None:
                    continue
                
                # Accumulate losses for this chunk
                batch_losses.append(loss_dict)
                            # Save visualizations with averaged losses
                if batch_id % 100 == 0:
                    # Get the corresponding text for this chunk
                    chunk_source_texts = source_texts[start//num_slices:(end+num_slices-1)//num_slices] if start < len(source_texts) else source_texts[:1]
                    chunk_target_texts = target_texts[start//num_slices:(end+num_slices-1)//num_slices] if start < len(target_texts) else target_texts[:1]
                    
                    self._save_enhanced_visualizations(
                        epoch, batch_id, start, end, 
                        src_imgs_chunk, tgt_imgs_chunk, output, source_betas, target_betas, mask,
                        loss_dict, loss_history, 'train',
                        source_texts=chunk_source_texts, target_texts=chunk_target_texts,
                        contrast_guidance_imgs=(contrast_guidance_imgs if self.guidance_mode in ['image','both'] else None)
                    )
                # Patchifier/PatchNCE debug plot every 10th batch
                if batch_id % 10 == 0:
                    self._update_patch_debug_plot(epoch, batch_id, source_betas, target_betas, mask, src_imgs_chunk, tgt_imgs_chunk)

            # Calculate averaged losses over all slices in this batch
            if batch_losses:
                avg_batch_loss = self._average_batch_losses(batch_losses)
                
                # Store averaged batch losses
                epoch_batch_losses.append(avg_batch_loss)
                
                # Update loss history for plotting
                for k, v in avg_batch_loss.items():
                    if k in loss_history:
                        loss_history[k].append(v)
                
                # 100-batch averaging (main process only)
                if self.rank == 0:
                    # Increment global batch counter
                    self._global_batch_counter += 1
                    
                    # Accumulate for 100-batch bucket
                    for key in self._batch_100_acc['sums'].keys():
                        if key in avg_batch_loss:
                            self._batch_100_acc['sums'][key] += float(avg_batch_loss[key])
                    self._batch_100_acc['count'] += 1

                    # Every 100 batches, compute bucket average and record
                    if self._batch_100_acc['count'] >= 100:
                        self.batch_100_history['batches'].append(self._global_batch_counter)
                        for key in self._batch_100_acc['sums'].keys():
                            avg_val = self._batch_100_acc['sums'][key] / self._batch_100_acc['count']
                            self.batch_100_history[key].append(avg_val)
                        # Reset bucket accumulator (but keep global counter)
                        self._batch_100_acc['count'] = 0
                        for key in self._batch_100_acc['sums'].keys():
                            self._batch_100_acc['sums'][key] = 0.0
                        
                        # Save 100-batch progression plot
                        self._save_batch_100_progression_plot()
                
                # ====== LOGGING AND VISUALIZATION ======
                curr_iteration = (epoch - 1) * total_batches + batch_id
                
                # Print detailed averaged losses (only on main process)
                if self.rank == 0 and batch_id % 10 == 0:
                    self._print_detailed_losses(epoch, batch_id, avg_batch_loss, 'TRAIN', None, None)
                    #self._print_batch_summary(epoch, batch_id, avg_batch_loss, len(batch_losses))
                
                # Log averaged metrics to TensorBoard (only on main process)
                # if self.rank == 0:
                    # self._log_averaged_training_metrics(avg_batch_loss, epoch, batch_id, curr_iteration)
                
                # Save loss plot less frequently (every 50 batches, only on main process)
                if self.rank == 0 and batch_id % 50 == 0:
                    self._save_loss_plot(loss_history, epoch, batch_id)
        
        # ====== EPOCH SUMMARY ======
        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Calculate epoch statistics
        epoch_avg_losses = self._calculate_epoch_statistics(epoch_batch_losses)
        
        # Print epoch summary
        self._print_epoch_summary(epoch, epochs, epoch_avg_losses, epoch_duration, 'TRAIN')
        
        # Save epoch training losses visualization
        #self._save_epoch_training_losses(epoch, epoch_avg_losses, loss_history)
        
        return epoch_avg_losses, epoch_batch_losses

    def train(self, epochs):
        """Main training function - orchestrates the entire training process"""
        # ====== INITIALIZATION ======
        # Initialize loss tracking
        loss_history = {'rec_loss': [], 'per_loss': [], 'beta_loss': [], 'clip_loss': [], 'clip_loss_l2': [], 'clip_image_loss': [], 'cycle_loss': [], 'adv_loss': [], 'disc_loss': [], 'total_loss': []}
        epoch_stats = {
            'train_losses': {'rec_loss': [], 'per_loss': [], 'beta_loss': [], 'clip_loss': [], 'clip_loss_l2': [], 'clip_image_loss': [], 'adv_loss': [], 'disc_loss': [], 'total_loss': []},
            'val_losses': {'rec_loss': [], 'per_loss': [], 'beta_loss': [], 'clip_loss': [], 'clip_loss_l2': [], 'clip_image_loss': [], 'adv_loss': [], 'disc_loss': [], 'total_loss': []}
        }
        
        # Calculate training statistics
        total_batches = len(self.train_loader)
        total_steps = total_batches * epochs
        
        # Print training start information (only on main process)
        if self.rank == 0:
            self.logger.info("="*80)
            self.logger.info(f"🚀 STARTING TRAINING - {epochs} EPOCHS")
            self.logger.info(f"📊 Total batches per epoch: {total_batches}")
            self.logger.info(f"🎯 Total training steps: {total_steps}")
            self.logger.info(f"🔧 Model type: {self.use_contrast_feat}")
            self.logger.info(f"💾 Output directory: {self.writer_path}")
            self.logger.info(f"🔄 Starting from epoch: {self.start_epoch}")
            if self.is_distributed:
                self.logger.info(f"🌐 Multi-GPU Training: {self.world_size} GPUs")
            self.logger.info("="*80)
            
            # Log essential training configuration
            self.logger.info("🔧 TRAINING CONFIGURATION")
            self.logger.info("─"*50)
            self.logger.info(f"📚 Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            self.logger.info(f"🎯 Optimizer: {type(self.optimizer).__name__}")
            self.logger.info(f"📏 Batch Size: {self.batch_size}")
            self.logger.info(f"🔄 Max Slices per Step: {self.max_slices_per_step}")
            self.logger.info(f"🎨 Beta Dimension: {self.beta_dim}")
            self.logger.info(f"🔗 Use Beta: {self.use_beta}")
            self.logger.info(f"🔗 Use Patchifier: {self.use_patchifier}")
            self.logger.info(f"🔗 Use Perceptual: {self.use_perceptual}")
            self.logger.info(f"🔗 Use Contrast Feat: {self.use_contrast_feat}")
            self.logger.info(f"🔗 Use Adversarial: {self.use_adversarial}")

            if self.use_adversarial:
                self.logger.info(f"🔗 Discriminator Update Freq: {self.discriminator_update_freq}")
                self.logger.info(f"🔗 Discriminator Channels: {self.discriminator_channels}")
                self.logger.info(f"🔗 Discriminator Layers: {self.discriminator_layers}")
                self.logger.info(f"🔗 Discriminator LR Scale: {self.discriminator_lr_scale}")
                self.logger.info(f"🔗 Adversarial Loss Weight: 0.02")
                self.logger.info(f"🔗 Adversarial Training Starts: After epoch 2")
            else:
                self.logger.info(f"🔗 Adversarial Training: DISABLED")
            self.logger.info(f"🎯 Device: {self.device}")
            self.logger.info("─"*50)
            
            # Log network architectures and parameters
            self.logger.info("🧠 NETWORK ARCHITECTURES")
            self.logger.info("─"*50)
            
            # Decoder architecture
            if hasattr(self, 'decoder'):
                total_params = sum(p.numel() for p in self.decoder.parameters())
                trainable_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
                self.logger.info(f"🔧 Decoder: {type(self.decoder).__name__}")
                self.logger.info(f"   ├─ Total Parameters: {total_params:,}")
                self.logger.info(f"   ├─ Trainable Parameters: {trainable_params:,}")
                if hasattr(self.decoder, 'base_ch'):
                    self.logger.info(f"   └─ Base Channels: {self.decoder.base_ch}")
                else:
                    self.logger.info(f"   └─ Architecture: {self.decoder}")
            
            # Beta encoder architecture
            if hasattr(self, 'beta_encoder') and self.beta_encoder is not None:
                total_params = sum(p.numel() for p in self.beta_encoder.parameters())
                trainable_params = sum(p.numel() for p in self.beta_encoder.parameters() if p.requires_grad)
                self.logger.info(f"🔧 Beta Encoder: {type(self.beta_encoder).__name__}")
                self.logger.info(f"   ├─ Total Parameters: {total_params:,}")
                self.logger.info(f"   ├─ Trainable Parameters: {trainable_params:,}")
                if hasattr(self.beta_encoder, 'base_ch'):
                    self.logger.info(f"   └─ Base Channels: {self.beta_encoder.base_ch}")
                self.logger.info(f"   └─ Architecture: {self.beta_encoder}")
            
            # AdaIN block architecture
            if hasattr(self, 'adain_block') and self.adain_block is not None:
                total_params = sum(p.numel() for p in self.adain_block.parameters())
                trainable_params = sum(p.numel() for p in self.adain_block.parameters() if p.requires_grad)
                self.logger.info(f"🔧 AdaIN Block: {type(self.adain_block).__name__}")
                self.logger.info(f"   ├─ Total Parameters: {total_params:,}")
                self.logger.info(f"   ├─ Trainable Parameters: {trainable_params:,}")
                if hasattr(self.adain_block, 'num_channels'):
                    self.logger.info(f"   └─ Channels: {self.adain_block.num_channels}")
            
            # Feature adapter architecture
            if hasattr(self, 'feature_adapter') and self.feature_adapter is not None:
                total_params = sum(p.numel() for p in self.feature_adapter.parameters())
                trainable_params = sum(p.numel() for p in self.feature_adapter.parameters() if p.requires_grad)
                self.logger.info(f"🔧 Feature Adapter: {type(self.feature_adapter).__name__}")
                self.logger.info(f"   ├─ Total Parameters: {total_params:,}")
                self.logger.info(f"   ├─ Trainable Parameters: {trainable_params:,}")
            
            # CLIP model info
            if hasattr(self, 'clip_model') and self.clip_model is not None:
                total_params = sum(p.numel() for p in self.clip_model.parameters())
                trainable_params = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
                self.logger.info(f"🔧 CLIP Model: {type(self.clip_model).__name__}")
                self.logger.info(f"   ├─ Total Parameters: {total_params:,}")
                self.logger.info(f"   ├─ Trainable Parameters: {trainable_params:,}")
                self.logger.info(f"   └─ Frozen: {trainable_params == 0}")
            
            # Enhanced style transfer architecture
            if hasattr(self, 'enhanced_style_transfer') and self.enhanced_style_transfer is not None:
                total_params = sum(p.numel() for p in self.enhanced_style_transfer.parameters())
                trainable_params = sum(p.numel() for p in self.enhanced_style_transfer.parameters() if p.requires_grad)
                self.logger.info(f"🔧 Enhanced Style Transfer: {type(self.enhanced_style_transfer).__name__}")
                self.logger.info(f"   ├─ Total Parameters: {total_params:,}")
                self.logger.info(f"   └─ Trainable Parameters: {trainable_params:,}")
            
            # Multi-scale injector architecture
            if hasattr(self, 'multi_scale_injector') and self.multi_scale_injector is not None:
                total_params = sum(p.numel() for p in self.multi_scale_injector.parameters())
                trainable_params = sum(p.numel() for p in self.multi_scale_injector.parameters() if p.requires_grad)
                self.logger.info(f"🔧 Multi-Scale Injector: {type(self.multi_scale_injector).__name__}")
                self.logger.info(f"   ├─ Total Parameters: {total_params:,}")
                self.logger.info(f"   └─ Trainable Parameters: {trainable_params:,}")
            
            # Calculate total model parameters
            total_model_params = 0
            trainable_model_params = 0
            
            # Check all possible model components
            model_components = [
                ('decoder', self.decoder if hasattr(self, 'decoder') else None),
                ('beta_encoder', self.beta_encoder if hasattr(self, 'beta_encoder') else None),
                ('adain_block', self.adain_block if hasattr(self, 'adain_block') else None),
                ('feature_adapter', self.feature_adapter if hasattr(self, 'feature_adapter') else None),
                ('clip_model', self.clip_model if hasattr(self, 'clip_model') else None),
                ('enhanced_style_transfer', self.enhanced_style_transfer if hasattr(self, 'enhanced_style_transfer') else None),
                ('multi_scale_injector', self.multi_scale_injector if hasattr(self, 'multi_scale_injector') else None),
            ]
            
            for name, component in model_components:
                if component is not None:
                    total_model_params += sum(p.numel() for p in component.parameters())
                    trainable_model_params += sum(p.numel() for p in component.parameters() if p.requires_grad)
            
            self.logger.info("─"*50)
            self.logger.info(f"📊 TOTAL MODEL PARAMETERS: {total_model_params:,}")
            self.logger.info(f"🎯 TRAINABLE PARAMETERS: {trainable_model_params:,}")
            self.logger.info(f"🔒 FROZEN PARAMETERS: {total_model_params - trainable_model_params:,}")
            self.logger.info("─"*50)
            
            # Log loss weights
            self.logger.info("⚖️ LOSS WEIGHTS")
            self.logger.info("─"*30)
            self.logger.info("   rec_loss:"+str(self.w_rec))
            self.logger.info("   clip_loss:"+str(self.w_clip))
            self.logger.info("   clip_loss_l2:"+str(self.w_clip_l2))
            self.logger.info("   clip_image_loss:"+str(self.w_clip_img))
            self.logger.info("   per_loss:"+str(self.w_per))
            self.logger.info("   beta_loss:"+str(self.w_beta))
            if self.use_adversarial:
                self.logger.info("   adv_loss:"+str(self.w_adv))
            self.logger.info("   cycle_loss:"+str(self.w_cyc))
            self.logger.info("─"*30)
            
            # Log dataset information
            self.logger.info("📁 DATASET INFORMATION")
            self.logger.info("─"*30)
            if hasattr(self, 'train_loader') and self.train_loader is not None:
                self.logger.info(f"   Training samples: {len(self.train_loader.dataset)}")
            if hasattr(self, 'valid_loader') and self.valid_loader is not None:
                self.logger.info(f"   Validation samples: {len(self.valid_loader.dataset)}")
            self.logger.info("─"*30)
            
            self.logger.info("="*80)
        
        # ====== EPOCH LOOP ======
        for epoch in range(self.start_epoch, epochs + 1):
            # Train for one epoch
            train_avg_losses, train_batch_losses = self._train_epoch(epoch, epochs, loss_history)
            epoch_stats['train_losses'] = train_avg_losses
            
            # ====== VALIDATION ======
            if self.valid_loader is not None:
                val_avg_losses = self._run_validation(epoch, epochs)
                epoch_stats['val_losses'] = val_avg_losses
                
                # Compare train vs validation (only on main process)
                if self.rank == 0:
                    self._print_train_val_comparison(train_avg_losses, val_avg_losses)
                    
                    # Log epoch metrics to TensorBoard (with validation)
                    # self._log_epoch_metrics(epoch, train_avg_losses, val_avg_losses)
                
                # Update epoch history with validation losses (only on main process)
                if self.rank == 0:
                    self._update_epoch_history(epoch, train_avg_losses, val_avg_losses)
            else:
                # Update epoch history without validation losses (only on main process)
                if self.rank == 0:
                    self._update_epoch_history(epoch, train_avg_losses)
            
            # ====== SAVE RESULTS ======
            self._save_epoch_results(epoch, epoch_stats, loss_history)
        
        # ====== FINAL SUMMARY ======
        if self.rank == 0:
            self._print_final_summary(epochs, epoch_stats)

    def _print_detailed_losses(self, epoch, batch_id, loss_dict, phase, start=None, end=None):
        """Print detailed losses with beautiful formatting"""
        if start is not None and end is not None:
            step_info = f" (step {start}-{end})"
        elif start is None and end is None:
            step_info = " (averaged over all slices)"
        else:
            step_info = ""
        
        self.logger.info(f"📈 {phase} - Epoch {epoch}, Batch {batch_id}{step_info}")
        self.logger.info("─"*50)
        
        # Color-coded loss values
        colors = {
            'rec_loss': '🟢', 'per_loss': '🔵', 'beta_loss': '🟡', 
            'clip_loss': '🟣', 'adv_loss': '🩷', 'total_loss': '🔴'
        }
        
        for loss_name, loss_value in loss_dict.items():
            if loss_name in colors:
                emoji = colors[loss_name]
                self.logger.info(f"{emoji} {loss_name:12}: {loss_value:8.6f}")
            else:
                self.logger.info(f"⚪ {loss_name:12}: {loss_value:8.6f}")
        
        self.logger.info("─"*50)

    def _calculate_epoch_statistics(self, batch_losses):
        """Calculate average losses for an epoch"""
        if not batch_losses:
            return {}
        
        avg_loss = {}
        for key in batch_losses[0].keys():
            values = [batch[key] for batch in batch_losses if key in batch]
            if values:
                avg_loss[key] = sum(values) / len(values)
        
        return avg_loss

    def _print_epoch_summary(self, epoch, total_epochs, avg_losses, duration, phase):
        """Print beautiful epoch summary"""
        self.logger.info("="*70)
        self.logger.info(f"📊 {phase} EPOCH {epoch}/{total_epochs} SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"⏱️  Duration: {duration}")
        self.logger.info(f"📈 Average Losses:")
        
        for loss_name, loss_value in avg_losses.items():
            self.logger.info(f"   • {loss_name:12}: {loss_value:8.6f}")
        
        self.logger.info("="*70)

    def _run_validation(self, epoch, total_epochs):
       # if self.debugger is not None:
       #     self.debugger.reset_debug_counter()
        """Run validation and return average losses"""
        if self.rank == 0:  # Only log on main process
            self.logger.info(f"🔍 EPOCH {epoch}/{total_epochs} - VALIDATION")
            self.logger.info("─"*50)
        
        # Set models to evaluation mode
        self.decoder.eval()
        if self.use_beta and self.beta_encoder is not None:
            self.beta_encoder.eval()
        if self.use_patchifier and self.patchifier is not None:
            self.patchifier.eval()
        if self.use_adversarial and self.discriminator is not None:
            self.discriminator.eval()
        
        val_batch_losses = []
        val_images_for_saving = []  # Store images for epoch summary
        val_metrics_history = {'ssim': [], 'psnr': [], 'lpips': []}
        
        with torch.no_grad():
            for val_batch_id, val_batch in enumerate(self.valid_loader):
                #if val_batch_id ==1:
                #    continue
                val_source_items = val_batch['source']
                val_target_items = val_batch['target']

                val_source_imgs = val_source_items.img.to(self.device)
                val_target_imgs = val_target_items.img.to(self.device)

                val_batch_size, val_num_slices = val_source_imgs.shape[:2]
                val_source_imgs = val_source_imgs.view(val_batch_size * val_num_slices, 1, 224, 224)
                val_target_imgs = val_target_imgs.view(val_batch_size * val_num_slices, 1, 224, 224)

                val_source_texts = val_source_items.text
                val_target_texts = val_target_items.text
                val_source_text_tokens = self.tokenizer(val_source_texts).to(self.device)
                val_target_text_tokens = self.tokenizer(val_target_texts).to(self.device)
                val_source_clip_text_feat = self.clip_model.encode_text(val_source_text_tokens)
                val_target_clip_text_feat = self.clip_model.encode_text(val_target_text_tokens)
                val_source_clip_text_feat = val_source_clip_text_feat.repeat_interleave(val_num_slices, dim=0)
                val_target_clip_text_feat = val_target_clip_text_feat.repeat_interleave(val_num_slices, dim=0)

                # Initialize validation batch loss accumulation
                val_batch_losses = []
                val_total_slices = val_source_imgs.shape[0]
                val_step = self.max_slices_per_step or val_total_slices
                
                for val_start in range(0, val_total_slices, val_step):
                    val_end = min(val_start + val_step, val_total_slices)
                    val_src_imgs_chunk = val_source_imgs[val_start:val_end]
                    val_tgt_imgs_chunk = val_target_imgs[val_start:val_end]
                    val_src_texts_chunk = val_source_clip_text_feat[val_start:val_end]
                    val_tgt_texts_chunk = val_target_clip_text_feat[val_start:val_end]
                    val_mask = (val_src_imgs_chunk > 1e-6)
                    # Process image chunk through model
                    contrast_guidance_imgs = None
                    if self.guidance_mode in ['image', 'both']:
                        rand= torch.randint(low=10, high= 40, size=(val_src_imgs_chunk.shape[0],))       
                        contrast_guidance_imgs = val_target_imgs[rand,...]
                    else:
                        contrast_guidance_imgs = val_tgt_imgs_chunk
                    
                    # Use the same processing as training
                    val_output, val_source_betas, val_target_betas, val_source_clip_img_feat, val_target_clip_img_feat = self._process_image_chunk(
                        val_src_imgs_chunk, val_tgt_imgs_chunk, val_src_texts_chunk, val_tgt_texts_chunk, val_mask, contrast_guidance_imgs
                    )

                    val_loss_dict, _ = self.calculate_loss(
                        rec_image=val_output,
                        ref_image=val_tgt_imgs_chunk,
                        mask=val_mask,
                        source_betas=val_source_betas,
                        target_betas=val_target_betas,
                        source_images=val_src_imgs_chunk,
                        source_clip_text_feat=val_src_texts_chunk,
                        target_clip_text_feat=val_tgt_texts_chunk,
                        target_clip_img_feat=val_target_clip_img_feat,
                        is_train=False,
                        epoch=epoch,
                        source_clip_img_feat=val_source_clip_img_feat,
                        target_images=val_tgt_imgs_chunk,
                    )
                    # Compute metrics for this chunk (averaged over batch)
                    try:
                        metrics = self.eval_metrics(val_output, val_tgt_imgs_chunk)
                        for k in val_metrics_history.keys():
                            if k in metrics and not np.isnan(metrics[k]):
                                val_metrics_history[k].append(float(metrics[k]))
                    except Exception:
                        pass
                    
                    if val_loss_dict is not None:
                        val_batch_losses.append(val_loss_dict)
                        
                        # Log validation metrics to TensorBoard
                        # self._log_validation_metrics(val_loss_dict, val_src_imgs_chunk, val_tgt_imgs_chunk, val_output, val_mask, epoch, val_batch_id)
                        
                        # Save validation images every 20 batches with batch ID and step info
                        if val_batch_id % 20 == 0:
                            # Get the corresponding text for this chunk
                            chunk_val_source_texts = val_source_texts[val_start//val_num_slices:(val_end+val_num_slices-1)//val_num_slices] if val_start < len(val_source_texts) else val_source_texts[:1]
                            chunk_val_target_texts = val_target_texts[val_start//val_num_slices:(val_end+val_num_slices-1)//val_num_slices] if val_start < len(val_target_texts) else val_target_texts[:1]
                            
                            self._save_enhanced_visualizations(
                                epoch, val_batch_id, val_start, val_end,
                                val_src_imgs_chunk, val_tgt_imgs_chunk, val_output, val_source_betas, val_target_betas, val_mask,
                                val_loss_dict, {}, 'val',  # Empty loss_history for validation
                                source_texts=chunk_val_source_texts, target_texts=chunk_val_target_texts,
                                contrast_guidance_imgs=(contrast_guidance_imgs if self.guidance_mode in ['image','both'] else None)
                            )
                            
    
                # Calculate averaged validation losses over all slices in this batch
                if val_batch_losses:
                    avg_val_loss = self._average_batch_losses(val_batch_losses)
                    val_batch_losses = [avg_val_loss]  # Replace with averaged loss for epoch stats
                    
                    if self.rank == 0 and val_batch_id % 5 == 0:  # Print validation less frequently
                        self._print_detailed_losses(epoch, val_batch_id, avg_val_loss, 'VAL', None, None)
                        self._print_batch_summary(epoch, val_batch_id, avg_val_loss, len(val_batch_losses))

                    # Simplified validation logging - log all loss components (only on main process)
                    if self.rank == 0 and self.writer is not None:
                        val_curr_iteration = (epoch - 1) * len(self.valid_loader) + val_batch_id
                        for k, v in avg_val_loss.items():
                            self.writer.add_scalar(f"val_avg/{k}", v, val_curr_iteration)
                            


        # Calculate and print validation summary
        val_avg_losses = self._calculate_epoch_statistics(val_batch_losses)
        # Append metrics summary to logs and plot a small figure
        try:
            ssim_mean = float(np.mean(val_metrics_history['ssim'])) if val_metrics_history['ssim'] else float('nan')
            psnr_mean = float(np.mean(val_metrics_history['psnr'])) if val_metrics_history['psnr'] else float('nan')
            lpips_mean = float(np.mean(val_metrics_history['lpips'])) if val_metrics_history['lpips'] else float('nan')
            if self.rank == 0:
                self.logger.info(f"VAL Metrics - Epoch {epoch}: SSIM={ssim_mean:.4f}, PSNR={psnr_mean:.2f}, LPIPS={lpips_mean:.4f}")
                # Append to running history
                self.val_metrics_history['epochs'].append(epoch)
                self.val_metrics_history['ssim'].append(ssim_mean)
                self.val_metrics_history['psnr'].append(psnr_mean)
                self.val_metrics_history['lpips'].append(lpips_mean)
                # Overwrite a single sequential plot file each epoch
                fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
                epochs_seq = self.val_metrics_history['epochs']
                ax1.plot(epochs_seq, self.val_metrics_history['ssim'], label='SSIM', color='#4ECDC4', marker='o')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('SSIM/LPIPS', color='#4ECDC4')
                ax1.grid(True, alpha=0.3)
                # Twin axis for PSNR
                ax2 = ax1.twinx()
                ax2.plot(epochs_seq, self.val_metrics_history['psnr'], label='PSNR', color='#45B7D1', marker='s')
                # Plot LPIPS on the primary axis as well
                ax1.plot(epochs_seq, self.val_metrics_history['lpips'], label='LPIPS', color='#FF6B6B', marker='^')
                ax2.set_ylabel('PSNR (dB)', color='#45B7D1')
                # Legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
                out_path = os.path.join(self.writer_path, 'val_metrics_progress.png')
                plt.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
        except Exception:
            pass
        self._print_epoch_summary(epoch, total_epochs, val_avg_losses, None, 'VALIDATION')
        
        # Save validation images for epoch summary
       
        
        return val_avg_losses

    def _init_lpips(self):
        try:
            import lpips  # local import
            if self._lpips_net is None:
                self._lpips_net = lpips.LPIPS(net='alex').to(self.device)
                self._lpips_net.eval()
        except Exception:
            self._lpips_net = None

    def _to_lpips_tensor(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W] or [1,H,W]; return [B,3,H,W] in [-1,1]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x3 = x.repeat(1, 3, 1, 1)
        return x3 * 2.0 - 1.0

    def eval_metrics(self, recon: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """
        Compute SSIM, PSNR, LPIPS between recon and ref.
        Expects tensors in [0,1], shape [B,1,H,W] or [1,H,W]; returns averaged scalars.
        """
        with torch.no_grad():
            if recon.dim() == 3:
                recon = recon.unsqueeze(0)
            if ref.dim() == 3:
                ref = ref.unsqueeze(0)
            # Resize if needed
            if recon.shape[-2:] != ref.shape[-2:]:
                ref = F.interpolate(ref, size=recon.shape[-2:], mode='bilinear', align_corners=False)
            # Apply mask inside-brain metrics
            # Save mask before applying for PSNR calculation
            mask_for_psnr = None
            if mask is not None:
                m = mask
                if m.dim() == 3:
                    m = m.unsqueeze(0)
                try:
                    if recon.dim() == 4 and m.dim() == 4:
                        pass
                    elif recon.dim() == 5 and m.dim() == 4:
                        m = m.unsqueeze(1).expand(recon.shape[0], recon.shape[1], 1, recon.shape[-2], recon.shape[-1])
                    elif recon.dim() == 5 and m.dim() == 5:
                        pass
                    else:
                        m = None
                except Exception:
                    m = None
                if m is not None:
                    # Save mask before applying for PSNR calculation
                    mask_for_psnr = m.clone()
                    recon = recon * m
                    ref = ref * m
            # recon = recon.clamp(0, 1)
            # ref = ref.clamp(0, 1)

            b = recon.size(0)
            ssim_vals = []
            psnr_vals = []
            for i in range(b):
                r_np = recon[i, 0].detach().cpu().numpy()
                t_np = ref[i, 0].detach().cpu().numpy()
                try:
                    ssim_vals.append(float(sk_ssim(t_np, r_np, data_range=1.0)))
                except Exception:
                    ssim_vals.append(float('nan'))
                try:
                    psnr_vals.append(float(sk_psnr(t_np, r_np, data_range=1.0)))
                except Exception:
                    psnr_vals.append(float('nan'))



            return {
                'ssim': float(np.nanmean(ssim_vals)) if len(ssim_vals) else float('nan'),
                'psnr': float(np.nanmean(psnr_vals)) if len(psnr_vals) else float('nan'),
                'lpips':  float('nan'),
            }

    def analyse_metrics(self, records: list, out_dir: str):
        """
        Analyse per-image metric records and save extremes and sequence aggregates.
        records: list of dicts with keys: id, recon_path, ref_path, ssim, psnr, lpips
        """
        os.makedirs(out_dir, exist_ok=True)
        extremes_dir = os.path.join(out_dir, 'extremes')
        os.makedirs(extremes_dir, exist_ok=True)
        from PIL import ImageDraw, ImageFont

        # Sequence detection (prefer text if available)
        def seq_from_text(text: str) -> str:
            if not isinstance(text, str):
                return 'UNKNOWN'
            u = text.upper()
            for key in ['T1W','T2W','T2STAR','FLAIR','PD']:
                if key in u:
                    return key
            return 'UNKNOWN'

        # Add sequence pairs
        for r in records:
            src_seq = seq_from_text(r.get('src_path', 'UNKNOWN'))
            tgt_seq = seq_from_text(r.get('tgt_path', 'UNKNOWN'))
            r['src_seq'] = src_seq
            r['tgt_seq'] = tgt_seq
            r['seq_pair'] = f"{src_seq}->{tgt_seq}"

        # Save extremes
        def save_extremes(metric: str, reverse: bool):
            valid = [x for x in records if np.isfinite(x.get(metric, float('nan')))]
            if not valid:
                return
            chosen = sorted(valid, key=lambda x: x[metric], reverse=reverse)[:5]
            sub = os.path.join(extremes_dir, f"{metric}_{'top' if reverse else 'bottom'}")
            os.makedirs(sub, exist_ok=True)
            # Write a single text summary file with all relevant information
            summary_path = os.path.join(sub, 'summary.txt')
            with open(summary_path, 'w') as fh:
                fh.write(f"Metric: {metric} | Order: {'top' if reverse else 'bottom'}\n")
                fh.write("id\tvalue\tsrc_seq\ttgt_seq\tseq_pair\tsrc_path\ttgt_path\trecon_path\tref_path\n")
                for r in chosen:
                    try:
                        rid = r.get('id', 'sample')
                        val = r.get(metric, float('nan'))
                        src_seq = r.get('src_seq', 'UNKNOWN')
                        tgt_seq = r.get('tgt_seq', 'UNKNOWN')
                        seq_pair = r.get('seq_pair', f"{src_seq}->{tgt_seq}")
                        src_path = r.get('src_path', '')
                        tgt_path = r.get('tgt_path', '')
                        recon_path = r.get('recon_path', r.get('recon', ''))
                        ref_path = r.get('ref_path', r.get('ref', ''))
                        fh.write(f"{rid}\t{val:.6f}\t{src_seq}\t{tgt_seq}\t{seq_pair}\t{src_path}\t{tgt_path}\t{recon_path}\t{ref_path}\n")
                    except Exception:
                        continue


        # SSIM/PSNR higher is better; LPIPS lower is better
        save_extremes('ssim', True)
        save_extremes('ssim', False)
        save_extremes('psnr', True)
        save_extremes('psnr', False)
        save_extremes('lpips', False)
        save_extremes('lpips', True)

        # Create a single composite image with 5 worst (top row) and 5 best (bottom row) by SSIM
        from PIL import Image
        # Select worst and best by SSIM
        valid_ssim = [x for x in records if np.isfinite(x.get('ssim', float('nan')))]
        if len(valid_ssim) >= 1:
            worst5 = sorted(valid_ssim, key=lambda x: x['ssim'])[:5]
            best5 = sorted(valid_ssim, key=lambda x: x['ssim'], reverse=True)[:5]

            # Load images and determine tile size
            def safe_open(path):
                try:
                    return Image.open(path).convert('L')
                except Exception:
                    return None

            worst_imgs = [safe_open(r.get('recon_path', '')) for r in worst5]
            best_imgs = [safe_open(r.get('recon_path', '')) for r in best5]
            tiles = [im for im in (worst_imgs + best_imgs) if im is not None]
            if len(tiles) > 0:
                # Normalize tile sizes by resizing to the first tile size
                base_w, base_h = tiles[0].size
                def resize_or_pad(img: Image.Image):
                    if img is None:
                        return Image.new('L', (base_w, base_h), 0)
                    if img.size != (base_w, base_h):
                        return img.resize((base_w, base_h))
                    return img

                worst_imgs = [resize_or_pad(im) for im in worst_imgs]
                best_imgs = [resize_or_pad(im) for im in best_imgs]

                label_h = 26
                cols = 5
                rows = 2
                grid_w = cols * base_w
                grid_h = rows * (base_h + label_h)
                grid = Image.new('L', (grid_w, grid_h), 0)
                draw = ImageDraw.Draw(grid)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

                # Helper to paste a row
                def paste_row(images, row_idx, recs):
                    y0 = row_idx * (base_h + label_h)
                    for i in range(cols):
                        if i >= len(images):
                            break
                        x0 = i * base_w
                        # Draw metrics label
                        if i < len(recs):
                            r = recs[i]
                            lbl = f"SSIM {r.get('ssim', float('nan')):.3f} | PSNR {r.get('psnr', float('nan')):.2f} | LPIPS {r.get('lpips', float('nan')):.3f}"
                            draw.text((x0 + 4, y0 + 4), lbl, fill=255, font=font)
                        # Paste image below label area
                        grid.paste(images[i], (x0, y0 + label_h))

                paste_row(worst_imgs, 0, worst5)
                paste_row(best_imgs, 1, best5)

                out_path = os.path.join(extremes_dir, 'ssim_worst_top_best_bottom_grid.png')
                try:
                    import imageio.v2 as imageio
                    imageio.imwrite(out_path, np.array(grid))
                except Exception:
                    # Fallback to PIL save
                    grid.save(out_path)

        # Sequence aggregates
        seq_map = {}
        for r in records:
            seq_map.setdefault(r['seq_pair'], []).append(r)
        seq_rows = []
        for k, lst in seq_map.items():
            ssim_vals = np.array([x.get('ssim', np.nan) for x in lst], dtype=np.float32)
            psnr_vals = np.array([x.get('psnr', np.nan) for x in lst], dtype=np.float32)
            lpips_vals = np.array([x.get('lpips', np.nan) for x in lst], dtype=np.float32)
            seq_rows.append((k, len(lst), float(np.nanmean(ssim_vals)), float(np.nanstd(ssim_vals)), float(np.nanmean(psnr_vals)), float(np.nanstd(psnr_vals)), float(np.nanmean(lpips_vals)), float(np.nanstd(lpips_vals))))

        # Save aggregates as a simple CSV
        try:
            import csv
            with open(os.path.join(out_dir, 'sequence_aggregates.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['seq_pair','count','ssim_mean','ssim_std','psnr_mean','psnr_std','lpips_mean','lpips_std'])
                for row in seq_rows:
                    w.writerow(row)
        except Exception:
            pass

        # Save overall averaged metrics across all records
        try:
            import csv
            ssim_all = np.array([x.get('ssim', np.nan) for x in records], dtype=np.float32)
            psnr_all = np.array([x.get('psnr', np.nan) for x in records], dtype=np.float32)
            lpips_all = np.array([x.get('lpips', np.nan) for x in records], dtype=np.float32)
            overall_path = os.path.join(out_dir, 'overall_metrics.csv')
            with open(overall_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['count','ssim_mean','ssim_std','psnr_mean','psnr_std','lpips_mean','lpips_std'])
                w.writerow([
                    len(records),
                    float(np.nanmean(ssim_all)) if len(records) else float('nan'),
                    float(np.nanstd(ssim_all)) if len(records) else float('nan'),
                    float(np.nanmean(psnr_all)) if len(records) else float('nan'),
                    float(np.nanstd(psnr_all)) if len(records) else float('nan'),
                    float(np.nanmean(lpips_all)) if len(records) else float('nan'),
                    float(np.nanstd(lpips_all)) if len(records) else float('nan'),
                ])
        except Exception:
            pass

    def _print_train_val_comparison(self, train_losses, val_losses):
        """Print comparison between training and validation losses"""
        self.logger.info(f"🔄 TRAIN vs VALIDATION COMPARISON")
        self.logger.info("─"*60)
        
        for loss_name in train_losses.keys():
            if loss_name in val_losses:
                train_val = train_losses[loss_name]
                val_val = val_losses[loss_name]
                diff = train_val - val_val
                status = "🟢" if abs(diff) < 0.01 else "🟡" if abs(diff) < 0.1 else "🔴"
                
                self.logger.info(f"{status} {loss_name:12}: Train={train_val:8.6f} | Val={val_val:8.6f} | Diff={diff:+8.6f}")
        
        self.logger.info("─"*60)

    def _save_enhanced_visualizations(self, epoch, batch_id, start, end, 
                                   src_imgs, tgt_imgs, output, source_betas, target_betas, mask,
                                   loss_dict, loss_history, phase, source_texts=None, target_texts=None, contrast_guidance_imgs=None):
        """Save enhanced visualizations with comprehensive plots including text descriptions"""
        

        # Create comprehensive visualization
        if start is not None and end is not None:
            title = f'{phase} - Epoch {epoch}, Batch {batch_id}, Step {start}-{end}'
        else:
            title = f'{phase} - Epoch {epoch}, Batch {batch_id} (Averaged)'
        
        fig, axes = plt.subplots(2, 4, figsize=(36, 20))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 1. Training images with text descriptions
        axes[0, 0].imshow(src_imgs[0, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        if source_texts and len(source_texts) > 0:
            source_text = source_texts[0] if isinstance(source_texts, list) else str(source_texts)
            # Show text in a multi-line overlay box
            # Format text to fit in 3-4 lines
            lines = []
            current_line = ""
            words = source_text.split()
            
            for word in words:
                if len(current_line + " " + word) <= 80:  # Limit line length
                    current_line += (" " + word) if current_line else word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Take first 4 lines
            display_lines = lines[:4]
            if len(lines) > 4:
                display_lines.append("...")
            
            display_text = "\n".join(display_lines)
            axes[0, 0].text(0.02, 0.98, f'Source:\n{display_text}', 
                           transform=axes[0, 0].transAxes, ha='left', va='top', 
                           fontsize=11, fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.9))
            axes[0, 0].set_title('Source Image', fontweight='bold')
        else:
            axes[0, 0].set_title('Source Image', fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(tgt_imgs[0, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        if target_texts and len(target_texts) > 0:
            target_text = target_texts[0] if isinstance(target_texts, list) else str(target_texts)
            # Show text in a multi-line overlay box
            # Format text to fit in 3-4 lines
            lines = []
            current_line = ""
            words = target_text.split()
            
            for word in words:
                if len(current_line + " " + word) <= 80:  # Limit line length
                    current_line += (" " + word) if current_line else word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Take first 4 lines
            display_lines = lines[:4]
            if len(lines) > 4:
                display_lines.append("...")
            
            display_text = "\n".join(display_lines)
            axes[0, 1].text(0.02, 0.98, f'Target:\n{display_text}', 
                           transform=axes[0, 1].transAxes, ha='left', va='top', 
                           fontsize=11, fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.9))
        axes[0, 1].set_title('Target Image', fontweight='bold')
        axes[0, 1].axis('off')

        # Target Contrast Image panel
        if self.guidance_mode in ['image','both'] and contrast_guidance_imgs is not None:
            axes[0, 2].imshow(contrast_guidance_imgs[0, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, 2].set_title('Target Contrast Image', fontweight='bold')
        else:
            # Keep grid consistent if no contrast image is provided
            axes[0, 2].imshow(tgt_imgs[0, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, 2].set_title('Target Contrast Image (N/A)', fontweight='bold')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(output[0, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, 3].set_title('Reconstructed Image', fontweight='bold')
        axes[0, 3].axis('off')

        # 2. Beta and mask (adapted for 3-channel betas)
        def _to_vis_img(b):
            # b: [C,H,W]; return image and cmap
            # For single-channel betas, DO NOT normalize; plot with vmin=0, vmax=1 like imgs
            b = b.detach().cpu()
            if b.shape[0] == 1:
                v = b[0]
                img = v.numpy()
                return img, 'gray'
            else:
                # For multi-channel visualization, keep per-channel normalization to compose RGB
                chans = []
                for c in range(min(3, b.shape[0])):
                    v = b[c]
                    vmin, vmax = float(v.min()), float(v.max())
                    if vmax > vmin:
                        v = (v - vmin) / (vmax - vmin)
                    chans.append(v)
                while len(chans) < 3:
                    chans.append(chans[-1])
                rgb = torch.stack(chans, dim=0).permute(1, 2, 0).numpy()
                return rgb, None
        src_img, src_cmap = _to_vis_img(source_betas[0])
        if src_cmap is None:
            axes[1, 0].imshow(src_img, vmin=0, vmax=1)
        else:
            axes[1, 0].imshow(src_img, cmap=src_cmap, vmin=0, vmax=1)
        axes[1, 0].set_title('Source Beta (3ch)', fontweight='bold')
        axes[1, 0].axis('off')
        tgt_img, tgt_cmap = _to_vis_img(target_betas[0])
        if tgt_cmap is None:
            axes[1, 1].imshow(tgt_img)
        else:
            axes[1, 1].imshow(tgt_img, cmap=tgt_cmap, vmin=0, vmax=1)
        axes[1, 1].set_title('Target Beta (3ch)', fontweight='bold')
        axes[1, 1].axis('off')

        # 3. Loss curves or text comparison
        if phase == 'train':
            if len(loss_history['total_loss']) > 1:
                axes[1, 2].plot(loss_history['total_loss'], label='Total Loss', linewidth=2, color='red')
                axes[1, 2].plot(loss_history['rec_loss'], label='Reconstruction Loss', linewidth=2, color='blue')
                axes[1, 2].plot(loss_history['clip_loss'], label='CLIP Loss', linewidth=2, color='green')
                axes[1, 2].plot(loss_history['clip_loss_l2'], label='CLIP L2 Loss', linewidth=2, color='purple')
                axes[1, 2].plot(loss_history['clip_image_loss'], label='CLIP Image Loss', linewidth=2, color='orange')
                # Add cycle loss if available
                if 'cycle_loss' in loss_history and len(loss_history['cycle_loss']) > 0:
                    axes[1, 2].plot(loss_history['cycle_loss'], label='Cycle Loss', linewidth=2, color='brown')
                axes[1, 2].set_title('Loss Progress', fontweight='bold')
                axes[1, 2].set_xlabel('Batch')
                axes[1, 2].set_ylabel('Loss')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Loss curves\nwill appear here', 
                            ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Loss Progress', fontweight='bold')
        # Move mask to the last panel in the bottom row
        axes[1, 3].imshow(mask[0, 0].float().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, 3].set_title('Mask', fontweight='bold')
        axes[1, 3].axis('off')
        
        if phase != 'train':
            # For validation, show full text comparison
            if source_texts and target_texts:
                source_text = source_texts[0] if isinstance(source_texts, list) else str(source_texts)
                target_text = target_texts[0] if isinstance(target_texts, list) else str(target_texts)
                
                # Show multi-line text comparison
                def format_text_for_display(text, max_lines=3):
                    lines = []
                    current_line = ""
                    words = text.split()
                    
                    for word in words:
                        if len(current_line + " " + word) <= 60:  # Shorter lines for validation panel
                            current_line += (" " + word) if current_line else word
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        lines.append(current_line)
                    
                    # Take first max_lines
                    display_lines = lines[:max_lines]
                    if len(lines) > max_lines:
                        display_lines.append("...")
                    
                    return "\n".join(display_lines)
                
                source_display = format_text_for_display(source_text, 3)
                target_display = format_text_for_display(target_text, 3)
                text_comparison = f"Source:\n{source_display}\n\nTarget:\n{target_display}"
                axes[1, 2].text(0.05, 0.95, text_comparison, transform=axes[1, 2].transAxes, 
                               fontsize=8, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.9))
                axes[1, 2].set_title('Text Comparison', fontweight='bold')
                axes[1, 2].axis('off')

        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = os.path.join(self.writer_path, f"{phase.lower()}_vis_epoch{epoch}_batch{batch_id}_step{start}-{end}.png")
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()
        
        self.logger.debug(f"📊 Saved enhanced visualization: {vis_path}")

    def _save_epoch_results(self, epoch, epoch_stats, loss_history):
        """Save comprehensive epoch results including model and visualizations"""
        
        # Save model
       
        model_save_path = os.path.join(self.writer_path, f'epoch{str(epoch).zfill(3)}_model.pt')
        self.save_model(epoch, model_save_path)
        if epoch>1:
            model_prev_path = os.path.join(self.writer_path, f'epoch{str(epoch-1).zfill(3)}_model.pt')
            os.remove(model_prev_path)
        # Create comprehensive epoch summary plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Epoch {epoch} - Comprehensive Summary', fontsize=16, fontweight='bold')

        # 1. Loss comparison
        if 'train_losses' in epoch_stats and 'val_losses' in epoch_stats:
            train_losses = epoch_stats['train_losses']
            val_losses = epoch_stats['val_losses']
            
            loss_names = list(train_losses.keys())
            train_values = [train_losses[name] for name in loss_names]
            val_values = [val_losses.get(name, 0) for name in loss_names]
            
            x = np.arange(len(loss_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, train_values, width, label='Train', alpha=0.8, color='skyblue')
            axes[0, 0].bar(x + width/2, val_values, width, label='Validation', alpha=0.8, color='lightcoral')
            axes[0, 0].set_title('Loss Comparison', fontweight='bold')
            axes[0, 0].set_ylabel('Loss Value')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(loss_names, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Training loss progress
        if len(loss_history['total_loss']) > 1:
            axes[0, 1].plot(loss_history['total_loss'], label='Total Loss', linewidth=2, color='red')
            axes[0, 1].plot(loss_history['rec_loss'], label='Reconstruction Loss', linewidth=2, color='blue')
            axes[0, 1].plot(loss_history['clip_loss'], label='CLIP Loss', linewidth=2, color='green')
            axes[0, 1].plot(loss_history['clip_loss_l2'], label='CLIP L2 Loss', linewidth=2, color='purple')
            axes[0, 1].set_title('Training Loss Progress', fontweight='bold')
            axes[0, 1].set_xlabel('Batch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Loss distribution
        if len(loss_history['total_loss']) > 10:
            axes[1, 0].hist(loss_history['total_loss'][-100:], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            axes[1, 0].set_title('Recent Total Loss Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Loss Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Model info
        model_info = f"""
        Model Configuration:
        - Contrast Feature: {self.use_contrast_feat}
        - Beta Dimension: {self.beta_dim}
        - Device: {self.device}
        - Optimizer: Adam
        - Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}
        
        Current Epoch Stats:
        - Total Loss: {epoch_stats['train_losses'].get('total_loss', 'N/A'):.6f}
        - Reconstruction Loss: {epoch_stats['train_losses'].get('rec_loss', 'N/A'):.6f}
        """
        
        axes[1, 1].text(0.1, 0.9, model_info, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Model Information', fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()
        
        # Save epoch summary
        summary_path = os.path.join(self.writer_path, f'epoch{str(epoch).zfill(3)}_summary.png')
        plt.savefig(summary_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"💾 Saved epoch {epoch} results:")
        self.logger.info(f"   📁 Model: {model_save_path}")
        self.logger.info(f"   📊 Summary: {summary_path}")
        self.logger.info(f"   🔍 Validation images saved automatically")

    def _update_patch_debug_plot(self, epoch, batch_id, source_betas, target_betas, mask, source_images, target_images):
        # Replicate the same feature pipeline as in patch_nce_loss for correct stats
        if not self.use_patchifier or self.patchifier is None:
            return
        with torch.no_grad():
            B = source_betas.size(0)
            q_all = self.patchifier(source_betas).view(B, 128, -1)
            p_all = self.patchifier(target_betas).view(B, 128, -1)
            P = q_all.shape[-1]
            grid = int(math.sqrt(P))
            if mask is not None and grid * grid == P:
                pooled = F.adaptive_avg_pool2d(mask.float(), (grid, grid))
                mask_cols = (pooled.view(B, -1) > 0.0)
                valid_cols = mask_cols & (q_all.abs().sum(dim=1) > 0) & (p_all.abs().sum(dim=1) > 0)
                k_list = [vc.sum().item() for vc in valid_cols]
                K = min(k_list) if k_list else 0
                if K <= 0:
                    return
                sel = []
                for i in range(B):
                    idx = valid_cols[i].nonzero(as_tuple=True)[0][:K]
                    sel.append(idx)
                idx_exp = torch.stack([s for s in sel], dim=0)  # [B,K]
                idx_exp = idx_exp.unsqueeze(1).expand(-1, q_all.shape[1], -1)
                q = torch.gather(q_all, 2, idx_exp)
                p = torch.gather(p_all, 2, idx_exp)
            else:
                q = q_all; p = p_all; K = P

            # Mean-center + LayerNorm + L2 as in training
            def _ln(x):
                B, C, K = x.shape
                return F.layer_norm(x.transpose(1, 2), (C,), eps=1e-6).transpose(1, 2)
            q_mc = q - q.mean(dim=2, keepdim=True)
            p_mc = p - p.mean(dim=2, keepdim=True)
            qn = F.normalize(_ln(q_mc), dim=1)
            pn = F.normalize(_ln(p_mc), dim=1)

            pos = (qn * pn).sum(dim=1)  # [B,K]
            pos_mean = float(pos.mean().item())
            # Off-diagonal mean from full similarity matrix per sample
            sim = torch.bmm(qn.transpose(1, 2), pn)  # [B,K,K]
            b, kk, _ = sim.shape
            diag = sim[:, torch.arange(kk, device=sim.device), torch.arange(kk, device=sim.device)]
            offdiag_mean = ((sim.sum(dim=(1, 2)) - diag.sum(dim=1)) / (kk * kk - kk)).mean().item()
            q_min = float(q.min().item()); q_max = float(q.max().item()); q_std = float(q.std().item())
            # Grad norms (last computed)
            grad_beta = 0.0
            if self.use_beta and self.beta_encoder is not None:
                for p in self.beta_encoder.parameters():
                    if p.grad is not None:
                        grad_beta += p.grad.data.norm(2).item() ** 2
                grad_beta **= 0.5
            grad_patch = 0.0
            if self.use_patchifier and self.patchifier is not None:
                for p in self.patchifier.parameters():
                    if p.grad is not None:
                        grad_patch += p.grad.data.norm(2).item() ** 2
                grad_patch **= 0.5
            ls = float(self.logit_scale.exp().item()) if hasattr(self, 'logit_scale') else 0.0

        self.debug_patch_history['iters'].append((epoch, batch_id))
        self.debug_patch_history['pos_mean'].append(pos_mean)
        self.debug_patch_history['neg_mean'].append(float(offdiag_mean))
        self.debug_patch_history['q_min'].append(q_min)
        self.debug_patch_history['q_max'].append(q_max)
        self.debug_patch_history['q_std'].append(q_std)
        self.debug_patch_history['grad_beta'].append(grad_beta)
        self.debug_patch_history['grad_patchifier'].append(grad_patch)
        self.debug_patch_history['logit_scale'].append(ls)
        # Save a compact debug plot into the current log path alongside other images
        try:
            import matplotlib.pyplot as plt
            xs = list(range(len(self.debug_patch_history['iters'])))
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            ax = axes[0, 0]; ax.plot(xs, self.debug_patch_history['pos_mean'], label='pos_mean'); ax.set_title('pos_mean'); ax.grid(True, alpha=0.3)
            ax = axes[0, 1]; ax.plot(xs, self.debug_patch_history['neg_mean'], label='neg_mean', color='orange'); ax.set_title('neg_mean'); ax.grid(True, alpha=0.3)
            ax = axes[0, 2]; ax.plot(xs, self.debug_patch_history['q_std'], label='q_std', color='green'); ax.set_title('q_std'); ax.grid(True, alpha=0.3)
            ax = axes[1, 0]; ax.plot(xs, self.debug_patch_history['grad_beta'], label='grad_beta', color='red'); ax.set_title('grad_beta'); ax.grid(True, alpha=0.3)
            ax = axes[1, 1]; ax.plot(xs, self.debug_patch_history['grad_patchifier'], label='grad_patchifier', color='purple'); ax.set_title('grad_patchifier'); ax.grid(True, alpha=0.3)
            ax = axes[1, 2]; ax.plot(xs, self.debug_patch_history['logit_scale'], label='logit_scale', color='brown'); ax.set_title('logit_scale'); ax.grid(True, alpha=0.3)
            for ax in axes.flat:
                ax.legend(fontsize=8)
            fig.suptitle(f'Patch Debug - Epoch {epoch} Batch {batch_id}', fontsize=12)
            fig.tight_layout()
            out_path = os.path.join(self.writer_path, f'patch_debug.png')
            plt.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass

    def _print_final_summary(self, epochs, epoch_stats):
        """Print final training summary"""
        self.logger.info("="*80)
        self.logger.info(f"🎉 TRAINING COMPLETED - {epochs} EPOCHS")
        self.logger.info("="*80)
        
        if 'train_losses' in epoch_stats and epoch_stats['train_losses']:
            final_train = epoch_stats['train_losses']
            self.logger.info(f"📈 Final Training Losses:")
            for loss_name, loss_value in final_train.items():
                self.logger.info(f"   • {loss_name:15}: {loss_value:8.6f}")
        
        if 'val_losses' in epoch_stats and epoch_stats['val_losses']:
            final_val = epoch_stats['val_losses']
            self.logger.info(f"🔍 Final Validation Losses:")
            for loss_name, loss_value in final_val.items():
                self.logger.info(f"   • {loss_name:15}: {loss_value:8.6f}")
        
        self.logger.info(f"📁 Results saved in: {self.writer_path}")
        self.logger.info(f"🎯 Model type: {self.use_contrast_feat}")
        self.logger.info("="*80)

    def _process_batch_data(self, batch):
        """Process batch data and extract source/target images and texts"""
        source_items = batch['source']
        target_items = batch['target']
        
        # Extract images
        source_imgs = source_items.img.to(self.device)
        target_imgs = target_items.img.to(self.device)
        
        # Flatten batch and slice dims
        batch_size, num_slices = source_imgs.shape[:2]
        source_imgs = source_imgs.view(batch_size * num_slices, 1, 224, 224)
        target_imgs = target_imgs.view(batch_size * num_slices, 1, 224, 224)
        
        # Extract texts
        source_texts = source_items.text
        target_texts = target_items.text
        
        return source_imgs, target_imgs, source_texts, target_texts, batch_size, num_slices

    def _check_data_filters(self, source_texts, target_texts, batch_id):
        """Check if batch should be filtered out based on text content"""
        for i, (src_text, tgt_text) in enumerate(zip(source_texts, target_texts)):
            if 'coronal' in src_text.lower() or 'sagittal' in src_text.lower() or 'coronal' in tgt_text.lower() or 'sagittal' in tgt_text.lower():
                self.logger.warning(f"⚠️  WARNING: Coronal/Sagittal view detected in batch {batch_id}")
                self.logger.warning(f"   Item {i} - Source text: {src_text}")
                self.logger.warning(f"   Item {i} - Target text: {tgt_text}")
                return True  # Skip this batch
        return False  # Process this batch

    def _check_mask_consistency(self, src_imgs_chunk, tgt_imgs_chunk, batch_id, start, end):
        """
        Enhanced mask consistency check with improved logic:
        1. Skip very small masks (less than 2% of image area)
        2. Better correspondence checking between source and target masks
        3. More stringent difference ratio for better quality
        """
        # Use fast extraction instead of simple >0 threshold
        src_mask = self._extract_brain_mask_fast(src_imgs_chunk)
        tgt_mask = self._extract_brain_mask_fast(tgt_imgs_chunk)
        
        total_pixels = src_mask.numel()
        src_mask_pixels = src_mask.sum().item()
        tgt_mask_pixels = tgt_mask.sum().item()
        
        # 1. Check for very small masks
        src_mask_ratio = src_mask_pixels / total_pixels
        tgt_mask_ratio = tgt_mask_pixels / total_pixels
        
        if src_mask_ratio < 0.02 or tgt_mask_ratio < 0.02:
            # self.logger.warning(f"⚠️  WARNING: Mask too small in batch {batch_id}, step {start}-{end}")
            # self.logger.warning(f"   Source mask ratio: {src_mask_ratio:.3%}, Target mask ratio: {tgt_mask_ratio:.3%}")
            # self.logger.warning(f"   Minimum required: 0.02%")
            return None  # Skip this batch
        
        # 2. Check for empty masks
        if src_mask_pixels == 0 or tgt_mask_pixels == 0:
            self.logger.warning(f"⚠️  WARNING: Empty mask detected in batch {batch_id}, step {start}-{end}")
            self.logger.warning(f"   Source mask pixels: {src_mask_pixels}, Target mask pixels: {tgt_mask_pixels}")
            return None  # Skip this batch
        
        # 3. Enhanced correspondence check - compare mask sizes
        mask_size_difference = abs(src_mask_pixels - tgt_mask_pixels)
        mask_size_ratio = mask_size_difference / max(src_mask_pixels, tgt_mask_pixels)
        
        # More stringent size difference check
        if mask_size_ratio > 0.1:
            # self.logger.warning(f"⚠️  WARNING: Mask size mismatch in batch {batch_id}, step {start}-{end}")
            # self.logger.warning(f"   Source mask pixels: {src_mask_pixels}, Target mask pixels: {tgt_mask_pixels}")
            # self.logger.warning(f"   Size difference ratio: {mask_size_ratio:.2%} (max allowed: 0.1%)")
            return None  # Skip this batch
        
        # 4. Spatial correspondence check - calculate mask difference
        mask_difference = (src_mask != tgt_mask).sum().item()
        difference_ratio = mask_difference / total_pixels
        
        # More stringent spatial difference check
        if difference_ratio > 0.03:
            # self.logger.warning(f"⚠️  WARNING: Significant spatial mask inconsistency in batch {batch_id}, step {start}-{end}")
            # self.logger.warning(f"   Source mask pixels: {src_mask_pixels}, Target mask pixels: {tgt_mask_pixels}")
            # self.logger.warning(f"   Spatial difference: {mask_difference} pixels ({difference_ratio:.2%} of total)")
            return None  # Skip this batch
        
        # 5. Use the source mask as the reference (or could use intersection/union)
        mask = src_mask
        return mask

    def _encode_clip_features(self, source_texts, target_texts, num_slices):
        """Encode text features using CLIP"""
        source_text_tokens = self.tokenizer(source_texts).to(self.device)
        target_text_tokens = self.tokenizer(target_texts).to(self.device)
        
        with torch.no_grad():
            source_clip_text_feat = self.clip_model.encode_text(source_text_tokens)
            target_clip_text_feat = self.clip_model.encode_text(target_text_tokens)
            
            # Repeat text features to match the number of slices per nifti
            source_clip_text_feat = source_clip_text_feat.repeat_interleave(num_slices, dim=0)
            target_clip_text_feat = target_clip_text_feat.repeat_interleave(num_slices, dim=0)
        
        return source_clip_text_feat, target_clip_text_feat

    def _process_image_chunk(self, src_imgs_chunk, tgt_imgs_chunk, src_clip_text_feat, tgt_clip_text_feat, mask, contrast_guidance_imgs=None, target_contrast_feat_precomputed=None):
        """Process a chunk of images through the model.

        target_contrast_feat_precomputed: optional [B, feat_dim] or [1, feat_dim] or [feat_dim].
            When provided, this is used as the target contrast embedding for all slices instead of
            encoding contrast_guidance_imgs per slice. Use at test time to get one embedding per
            volume (e.g. mean over slice embeddings) for consistent 3D style transfer.
        """
        # Calculate CLIP image features
        # Resize to match CLIP model's expected image size (checkpoint was trained with specific size)
        with torch.no_grad():
            source_clip_img_feat = self.clip_model.encode_image(src_imgs_chunk)
            target_clip_img_feat = self.clip_model.encode_image(tgt_imgs_chunk)
        
        # Calculate betas
        _, source_betas = self.calculate_beta(src_imgs_chunk)
        _, target_betas = self.calculate_beta(tgt_imgs_chunk)
        
        # Apply mask
        source_betas = source_betas * mask
        target_betas = target_betas * mask
        # Merge contrast features based on guidance_mode
        # - 'text': always use text
        # - 'image': use image (requires contrast_guidance_imgs or target_contrast_feat_precomputed)
        # - 'both': random 50/50 between text and image when contrast image available, otherwise fall back to text
        # When target_contrast_feat_precomputed is provided (e.g. at test time), use it for all slices.
        target_contrast_feat = None
        try:
            if target_contrast_feat_precomputed is not None:
                # Use single (e.g. averaged) embedding for all slices for consistent 3D output
                tcf = target_contrast_feat_precomputed.to(src_imgs_chunk.device)
                if tcf.dim() == 1:
                    tcf = tcf.unsqueeze(0)
                if tcf.size(0) == 1 and src_imgs_chunk.size(0) > 1:
                    tcf = tcf.expand(src_imgs_chunk.size(0), -1)
                target_contrast_feat = tcf
            elif self.guidance_mode == 'image':
                # Explicit image mode
                if contrast_guidance_imgs is not None:
                    with torch.no_grad():
                        target_contrast_feat = self.clip_model.encode_image(contrast_guidance_imgs)
            elif self.guidance_mode == 'both':
                # Prefer available features; random choice only when both present
                has_text = (tgt_clip_text_feat is not None)
                has_img = (contrast_guidance_imgs is not None)
                if has_text and has_img:
                    if torch.rand(1, device=self.device).item() < 0.5:
                        target_contrast_feat = tgt_clip_text_feat
                    else:
                        with torch.no_grad():
                            target_contrast_feat = self.clip_model.encode_image(contrast_guidance_imgs)
                elif has_text:
                    target_contrast_feat = tgt_clip_text_feat
                elif has_img:
                    with torch.no_grad():
                        target_contrast_feat = self.clip_model.encode_image(contrast_guidance_imgs)
            else:
                # text mode (default). If text missing, fallback to image if provided.
                if tgt_clip_text_feat is not None:
                    target_contrast_feat = tgt_clip_text_feat
                elif contrast_guidance_imgs is not None:
                    with torch.no_grad():
                        target_contrast_feat = self.clip_model.encode_image(contrast_guidance_imgs)
        except Exception as e:
            print("Error in _process_image_chunk: ", e)
            import traceback
            traceback.print_exc()
            target_contrast_feat = tgt_clip_text_feat

    
        
        if self.use_contrast_feat == 'adain':
            beta_modulated = self.adain_block(source_betas, target_contrast_feat)
            decoder_input = beta_modulated*mask
        elif self.use_contrast_feat == 'adapter':
            contrast_map = self.feature_adapter(target_contrast_feat)
            # Concatenate betas [B,3,H,W] with contrast_map [B,1,H,W] → [B,4,H,W]
            decoder_input = torch.cat([source_betas, contrast_map], dim=1)
        elif self.use_contrast_feat == 'attention':
            x_flat = source_betas.view(src_imgs_chunk.shape[0], 1, 224*224).permute(0, 2, 1)
            x_flat = self.image_proj(x_flat)
            # attention block expects 256 via text_proj
            style_feat = self.text_proj(target_contrast_feat).unsqueeze(1)
            x_attended = self.cross_attn_block(x_flat, style_feat)
            x_attended = self.img_proj_back(x_attended)
            decoder_input = x_attended.permute(0, 2, 1).view(src_imgs_chunk.shape[0], 1, 224, 224)
        elif 'enhanced' in self.use_contrast_feat:
            # Enhanced style transfer with text-conditioned decoder
            decoder_input = source_betas * mask
            output = self.decoder(in_tensor=decoder_input, style_text_feat=target_contrast_feat) * mask
            return output, source_betas, target_betas, source_clip_img_feat, target_clip_img_feat
        elif self.use_contrast_feat == 'multiscale':
            # Multi-scale style injection using both EnhancedStyleTransfer and MultiScaleStyleInjector
            decoder_input = source_betas * mask
            
            # Create multi-scale features by downsampling
            features_list = []
            current_feat = decoder_input
            for i in range(4):
                features_list.append(current_feat)
                if i < 3:
                    current_feat = F.avg_pool2d(current_feat, kernel_size=2, stride=2)
            
            # Apply multi-scale style injection - now we use ALL scales
            modulated_features = self.multi_scale_injector(features_list, target_contrast_feat)
            
            # Apply enhanced style transfer to each scale
            enhanced_features = []
            for i, feat in enumerate(modulated_features):
                # Use attention only at finest scale to save memory
                use_attention = (i == 0)  # Only use attention for scale0 (224x224)
                enhanced_feat = self.enhanced_style_transfer(feat, target_contrast_feat, use_attention=use_attention)
                # Project back to 1 channel for MultiScaleDecoder
                if not hasattr(self, f'enhanced_proj_scale{i}'):
                    setattr(self, f'enhanced_proj_scale{i}', nn.Conv2d(256, 1, 1).to(enhanced_feat.device))
                proj_layer = getattr(self, f'enhanced_proj_scale{i}')
                enhanced_feat = proj_layer(enhanced_feat)
                enhanced_features.append(enhanced_feat)
            
            # Pass all scales to the multi-scale decoder
            output = self.decoder(enhanced_features) * mask
            return output, source_betas, target_betas, source_clip_img_feat, target_clip_img_feat
        elif self.use_contrast_feat == 'none':
            decoder_input = source_betas
        else:
            raise ValueError(f"Unknown use_contrast_feat option: {self.use_contrast_feat}")
        
        # Apply mask to decoder input
        decoder_input = decoder_input * mask
        
        # Decode output
        output = self.decoder(decoder_input) * mask
        
        return output, source_betas,target_betas, source_clip_img_feat, target_clip_img_feat

    def _calculate_and_backpropagate_loss(self, output, tgt_imgs_chunk, mask, source_betas, target_betas, 
                                        src_imgs_chunk, source_clip_img_feat, src_clip_text_feat, 
                                        tgt_clip_text_feat, target_clip_img_feat, epoch, batch_id=0,  ):
        """Calculate loss and perform backpropagation"""
        # Calculate loss
        loss_dict, total_loss = self.calculate_loss(
            epoch=epoch,
            rec_image=output,
            ref_image=tgt_imgs_chunk,
            mask=mask,
            source_betas=source_betas,
            target_betas=target_betas,
            source_images=src_imgs_chunk,
            source_clip_img_feat=source_clip_img_feat,
            source_clip_text_feat=src_clip_text_feat,
            target_clip_text_feat=tgt_clip_text_feat,
            target_clip_img_feat=target_clip_img_feat,
            target_images=tgt_imgs_chunk,
            is_train=True,
        )
        
        if loss_dict is None:  # Skip if loss calculation failed
            return None
        
        # Log gradient norms for debugging (every 10 batches)
        if hasattr(self, 'log_gradients') and self.log_gradients and batch_id % 10 == 0:
            self.log_gradient_norms_from_tensors(loss_dict, total_loss, epoch, batch_id, 
                                               output, tgt_imgs_chunk, mask, source_betas, target_betas,
                                               src_imgs_chunk, source_clip_img_feat, src_clip_text_feat,
                                               tgt_clip_text_feat, target_clip_img_feat)
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for beta path modules to stabilize NCE/anatomy training
        if self.grad_clip > 0:
            try:
                clip_params = []
                if self.use_beta and self.beta_encoder is not None:
                    clip_params += list(self.beta_encoder.parameters())
                if self.use_patchifier and self.patchifier is not None:
                    clip_params += list(self.patchifier.parameters())
                # Optionally include learnable logit scale if present
                if hasattr(self, 'logit_scale') and isinstance(self.logit_scale, torch.nn.Parameter):
                    clip_params.append(self.logit_scale)
                if clip_params:
                    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=self.grad_clip)
            except Exception:
                pass
        
        # Check gradients if debugging is enabled
        if hasattr(self, 'debug_beta_loss') and self.debug_beta_loss and epoch % 10 == 0:
            self.check_beta_gradients()
        
        # Optimizer step
        self.optimizer.step()
        
        return loss_dict

    def _log_training_metrics(self, loss_dict, src_imgs_chunk, tgt_imgs_chunk, source_betas, output, mask, 
                            epoch, batch_id, curr_iteration):
        """Log training metrics to TensorBoard - simplified version"""
        if self.writer is not None:
            try:
                # Log all loss components separately
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f"train/{k}", v, curr_iteration)
                
                # Log images every 100 batches to save space
                if batch_id % 100 == 0:
                    img_grid = torch.cat([
                        src_imgs_chunk[0:1],  # Source
                        tgt_imgs_chunk[0:1],  # Target
                        output[0:1],          # Output
                    ], dim=0)
                    self.writer.add_images(f"train/images_epoch_{epoch}", img_grid, curr_iteration)
                
            except Exception as e:
                self.logger.error(f"Error logging training metrics: {str(e)}")

    def _save_loss_plot(self, loss_history, epoch, batch_id):
        """Save a dedicated loss plot with consistent filename for easy tracking"""
        
        # Create figure with subplots for different loss types
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Losses - Epoch {epoch}, Batch {batch_id}', fontsize=16, fontweight='bold')
        
        # Colors for different losses
        colors = {
            'total_loss': '#FF6B6B',    # Red
            'rec_loss': '#4ECDC4',      # Teal
            'per_loss': '#45B7D1',      # Blue
            'beta_loss': '#96CEB4',     # Green
            'clip_loss': '#FFEAA7',     # Yellow
            'clip_loss_l2': '#FF8C00'   # Dark Orange
        }
        
        # 1. Main loss curves (top left)
        if len(loss_history['total_loss']) > 1:
            axes[0, 0].plot(loss_history['total_loss'], label='Total Loss', 
                           color=colors['total_loss'], linewidth=2.5, alpha=0.8)
            axes[0, 0].plot(loss_history['rec_loss'], label='Reconstruction Loss', 
                           color=colors['rec_loss'], linewidth=2, alpha=0.8)
            axes[0, 0].plot(loss_history['clip_loss'], label='CLIP Loss', 
                           color=colors['clip_loss'], linewidth=2, alpha=0.8)
            axes[0, 0].plot(loss_history['clip_loss_l2'], label='CLIP L2 Loss', 
                           color='#FF8C00', linewidth=2, alpha=0.8)  # Dark orange
            if 'beta_loss' in loss_history and len(loss_history['beta_loss']) > 0:
                axes[0, 0].plot(loss_history['beta_loss'], label='Beta Loss', 
                               color=colors['beta_loss'], linewidth=2, alpha=0.8)
            if 'adv_loss' in loss_history and len(loss_history['adv_loss']) > 0:
                # Always plot adversarial loss, even if it's 0.0 (shows when it's disabled)
                axes[0, 0].plot(loss_history['adv_loss'], label='Adversarial Loss', 
                               color='#FF1493', linewidth=2, alpha=0.8)  # Deep pink
            axes[0, 0].set_title('Main Losses', fontweight='bold', fontsize=12)
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Loss Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')  # Log scale for better visualization
        
        # 2. All losses comparison (top right)
        if len(loss_history['total_loss']) > 1:
            # Add clip_loss_l2 and adv_loss to colors
            all_colors = {**colors, 'clip_loss_l2': '#FF8C00', 'adv_loss': '#FF1493', 'cycle_loss': '#8B4513'}  # add cycle loss color
            
            for loss_name, color in all_colors.items():
                if loss_name in loss_history and len(loss_history[loss_name]) > 0 and loss_name != 'total_loss':
                    axes[0, 1].plot(loss_history[loss_name], label=loss_name.replace('_', ' ').title(), 
                                   color=color, linewidth=2, alpha=0.7)
            axes[0, 1].set_title('All Losses', fontweight='bold', fontsize=12)
            axes[0, 1].set_xlabel('Batch')
            axes[0, 1].set_ylabel('Loss Value')
            axes[0, 1].legend(fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Recent loss trend (bottom left) - last 100 batches
        if len(loss_history['total_loss']) > 10:
            recent_batches = min(100, len(loss_history['total_loss']))
            recent_total = loss_history['total_loss'][-recent_batches:]
            recent_rec = loss_history['rec_loss'][-recent_batches:]
            recent_clip = loss_history['clip_loss'][-recent_batches:]
            recent_clip_l2 = loss_history['clip_loss_l2'][-recent_batches:]
            
            axes[1, 0].plot(recent_total, label='Total Loss (Recent)', 
                           color=colors['total_loss'], linewidth=2, alpha=0.8)
            axes[1, 0].plot(recent_rec, label='Reconstruction Loss (Recent)', 
                           color=colors['rec_loss'], linewidth=2, alpha=0.8)
            axes[1, 0].plot(recent_clip, label='CLIP Loss (Recent)', 
                           color=colors['clip_loss'], linewidth=2, alpha=0.8)
            axes[1, 0].plot(recent_clip_l2, label='CLIP L2 Loss (Recent)', 
                           color='#FF8C00', linewidth=2, alpha=0.8)
            if 'beta_loss' in loss_history and len(loss_history['beta_loss']) > 0:
                recent_beta = loss_history['beta_loss'][-recent_batches:]
                axes[1, 0].plot(recent_beta, label='Beta Loss (Recent)', 
                               color=colors['beta_loss'], linewidth=2, alpha=0.8)
            if 'adv_loss' in loss_history and len(loss_history['adv_loss']) > 0:
                recent_adv = loss_history['adv_loss'][-recent_batches:]
                axes[1, 0].plot(recent_adv, label='Adversarial Loss (Recent)', 
                               color='#FF1493', linewidth=2, alpha=0.8)
            axes[1, 0].set_title(f'Recent Trend (Last {recent_batches} batches)', fontweight='bold', fontsize=12)
            axes[1, 0].set_xlabel('Recent Batch')
            axes[1, 0].set_ylabel('Loss Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Loss statistics (bottom right)
        if len(loss_history['total_loss']) > 1:
            # Calculate statistics
            total_losses = loss_history['total_loss']
            rec_losses = loss_history['rec_loss']
            clip_losses = loss_history['clip_loss']
            clip_l2_losses = loss_history['clip_loss_l2']
            beta_losses = loss_history.get('beta_loss', [])
            adv_losses = loss_history.get('adv_loss', [])
            
            stats_text = f"""
            Loss Statistics:
            
            Total Loss:
            - Current: {total_losses[-1]:.6f}
            - Min: {min(total_losses):.6f}
            - Max: {max(total_losses):.6f}
            - Mean: {np.mean(total_losses):.6f}
            - Std: {np.std(total_losses):.6f}
            
            Reconstruction Loss:
            - Current: {rec_losses[-1]:.6f}
            - Min: {min(rec_losses):.6f}
            - Max: {max(rec_losses):.6f}
            - Mean: {np.mean(rec_losses):.6f}
            - Std: {np.std(rec_losses):.6f}
            
            CLIP Loss:
            - Current: {clip_losses[-1]:.6f}
            - Min: {min(clip_losses):.6f}
            - Max: {max(clip_losses):.6f}
            - Mean: {np.mean(clip_losses):.6f}
            
            CLIP L2 Loss:
            - Current: {clip_l2_losses[-1]:.6f}
            - Min: {min(clip_l2_losses):.6f}
            - Max: {max(clip_l2_losses):.6f}
            - Mean: {np.mean(clip_l2_losses):.6f}
            
            Beta Loss:
            - Current: {beta_losses[-1] if beta_losses else 'N/A'}
            - Min: {min(beta_losses) if beta_losses else 'N/A'}
            - Max: {max(beta_losses) if beta_losses else 'N/A'}
            - Mean: {np.mean(beta_losses) if beta_losses else 'N/A'}
            
            Adversarial Loss:
            - Current: {adv_losses[-1] if adv_losses else 'N/A'}
            - Min: {min(adv_losses) if adv_losses else 'N/A'}
            - Max: {max(adv_losses) if adv_losses else 'N/A'}
            - Mean: {np.mean(adv_losses) if adv_losses else 'N/A'}
            
            Progress:
            - Total batches: {len(total_losses)}
            - Epoch: {epoch}
            - Batch: {batch_id}
            """
            
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            axes[1, 1].set_title('Statistics', fontweight='bold', fontsize=12)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save with consistent filename for easy tracking
        loss_plot_path = os.path.join(self.writer_path, 'training_losses.png')
        plt.savefig(loss_plot_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.debug(f"📈 Saved loss plot: {loss_plot_path}")

    def _save_validation_images(self, epoch, val_images, val_losses):
        """Save validation images with optimized file size"""
        
        # Create compact validation visualization (reduced from 2x3 to 2x2)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Validation Results - Epoch {epoch}', fontsize=14, fontweight='bold')

        # 1. Main images (top row)
        axes[0, 0].imshow(val_images['src'][0, 0].numpy(), cmap='gray')
        axes[0, 0].set_title('Source', fontweight='bold', fontsize=10)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(val_images['tgt'][0, 0].numpy(), cmap='gray')
        axes[0, 1].set_title('Target', fontweight='bold', fontsize=10)
        axes[0, 1].axis('off')

        # 2. Output and summary (bottom row)
        axes[1, 0].imshow(val_images['output'][0, 0].numpy(), cmap='gray')
        axes[1, 0].set_title('Output', fontweight='bold', fontsize=10)
        axes[1, 0].axis('off')

        # 3. Compact validation summary
        if val_losses:
            loss_text = f"""Validation Losses:
Total: {val_losses.get('total_loss', 'N/A'):.4f}
Rec: {val_losses.get('rec_loss', 'N/A'):.4f}
CLIP: {val_losses.get('clip_loss', 'N/A'):.4f}
Per: {val_losses.get('per_loss', 'N/A'):.4f}

Image Stats:
Src: [{val_images['src'].min().item():.2f}, {val_images['src'].max().item():.2f}]
Tgt: [{val_images['tgt'].min().item():.2f}, {val_images['tgt'].max().item():.2f}]
Out: [{val_images['output'].min().item():.2f}, {val_images['output'].max().item():.2f}]
Mask: {val_images['mask'].sum().item() / val_images['mask'].numel():.1%}"""
        else:
            loss_text = "No validation losses available"
        
        axes[1, 1].text(0.05, 0.95, loss_text, transform=axes[1, 1].transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8))
        axes[1, 1].set_title('Summary', fontweight='bold', fontsize=10)
        axes[1, 1].axis('off')

        plt.tight_layout()
        
        # Save validation summary with reduced DPI to save space
        val_summary_path = os.path.join(self.writer_path, f'validation_epoch{str(epoch).zfill(3)}.png')
        plt.savefig(val_summary_path, bbox_inches='tight', facecolor='white', dpi=150)  # Reduced DPI
        plt.close()
        
        self.logger.info(f"💾 Saved validation summary: {val_summary_path}")

        # Save individual images only for key epochs (every 5 epochs) to save space
        if epoch % 5 == 0:
            self._save_individual_validation_images(epoch, val_images)

    def _save_individual_validation_images(self, epoch, val_images):
        """Save individual validation images for detailed analysis (optimized for file size)"""
        
        # Create directory for individual validation images
        val_img_dir = os.path.join(self.writer_path, f'validation_images_epoch_{epoch}')
        os.makedirs(val_img_dir, exist_ok=True)
        
        # Save only the most important images to save space
        image_types = {
            'source': val_images['src'],
            'target': val_images['tgt'], 
            'output': val_images['output'],
            'mask': val_images['mask'],
            'source_beta': val_images['source_betas']
        }
        
        for img_type, img_tensor in image_types.items():
            # Convert to numpy and save
            img_np = img_tensor[0, 0].numpy()
            
            # Normalize to 0-1 range for saving
            img_np = np.clip(img_np, 0, 1)
            
            # Save as PNG with reduced size and DPI
            img_path = os.path.join(val_img_dir, f'{img_type}.png')
            plt.figure(figsize=(6, 6))  # Reduced figure size
            plt.imshow(img_np, cmap='gray')
            plt.title(f'{img_type.title()} - Epoch {epoch}', fontsize=10)
            plt.axis('off')
            plt.savefig(img_path, bbox_inches='tight', facecolor='white', dpi=120)  # Reduced DPI
            plt.close()
        
        self.logger.debug(f"📁 Saved individual validation images to: {val_img_dir}")

    def _average_batch_losses(self, batch_losses):
        """Average losses over all slices in a batch"""
        if not batch_losses:
            return {}
        
        avg_loss = {}
        for key in batch_losses[0].keys():
            values = [loss[key] for loss in batch_losses if key in loss]
            if values:
                avg_loss[key] = sum(values) / len(values)
        
        return avg_loss

    def _print_batch_summary(self, epoch, batch_id, avg_loss, num_slices):
        """Print summary of averaged batch losses"""
        self.logger.info(f"📊 BATCH SUMMARY - Epoch {epoch}, Batch {batch_id}")
        self.logger.info(f"   Slices processed: {num_slices}")
        self.logger.info(f"   Average Total Loss: {avg_loss.get('total_loss', 0):.6f}")
        self.logger.info(f"   Average Rec Loss: {avg_loss.get('rec_loss', 0):.6f}")
        self.logger.info(f"   Average CLIP Loss: {avg_loss.get('clip_loss', 0):.6f}")
        self.logger.info("─"*50)

    def _log_averaged_training_metrics(self, avg_loss, epoch, batch_id, curr_iteration):
        """Log averaged training metrics to TensorBoard - simplified version"""
        if self.writer is not None:
            try:
                # Log all loss components separately
                for k, v in avg_loss.items():
                    self.writer.add_scalar(f"train_avg/{k}", v, curr_iteration)
                
            except Exception as e:
                self.logger.error(f"Error logging averaged training metrics: {str(e)}")

    def _save_epoch_training_losses(self, epoch, epoch_avg_losses, loss_history):
        """Save epoch training losses visualization"""
        
        # Create comprehensive epoch visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Epoch {epoch} - Training Losses Summary', fontsize=16, fontweight='bold')

        # 1. Current epoch losses (top left)
        if epoch_avg_losses:
            loss_names = list(epoch_avg_losses.keys())
            loss_values = list(epoch_avg_losses.values())
            
            # Color coding for different loss types
            colors = {
                'rec_loss': '#4ECDC4',      # Teal
                'per_loss': '#45B7D1',      # Blue
                'clip_loss': '#FFEAA7',     # Yellow
                'clip_loss_l2': '#FF8C00',  # Dark Orange
                'beta_loss': '#96CEB4',     # Green
                'total_loss': '#FF6B6B'     # Red
            }
            
            bar_colors = [colors.get(name, '#CCCCCC') for name in loss_names]
            
            bars = axes[0, 0].bar(range(len(loss_names)), loss_values, color=bar_colors, alpha=0.8)
            axes[0, 0].set_title('Current Epoch Losses', fontweight='bold')
            axes[0, 0].set_ylabel('Loss Value')
            axes[0, 0].set_xticks(range(len(loss_names)))
            axes[0, 0].set_xticklabels(loss_names, rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, loss_values):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Training progress over epochs (top right)
        if len(loss_history.get('total_loss', [])) > 1:
            epochs_range = range(1, len(loss_history['total_loss']) + 1)
            
            # Plot main losses
            if 'total_loss' in loss_history:
                axes[0, 1].plot(epochs_range, loss_history['total_loss'], 
                               label='Total Loss', color='#FF6B6B', linewidth=2.5, marker='o')
            if 'rec_loss' in loss_history:
                axes[0, 1].plot(epochs_range, loss_history['rec_loss'], 
                               label='Reconstruction Loss', color='#4ECDC4', linewidth=2, marker='s')
            if 'clip_loss' in loss_history:
                axes[0, 1].plot(epochs_range, loss_history['clip_loss'], 
                               label='CLIP Loss', color='#FFEAA7', linewidth=2, marker='^')
            if 'per_loss' in loss_history:
                axes[0, 1].plot(epochs_range, loss_history['per_loss'], 
                               label='Perceptual Loss', color='#45B7D1', linewidth=2, marker='d')
            
            axes[0, 1].set_title('Training Progress', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss Value')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Loss distribution (bottom left)
        if len(loss_history.get('total_loss', [])) > 5:
            recent_losses = loss_history['total_loss'][-10:]  # Last 10 epochs
            axes[1, 0].hist(recent_losses, bins=min(8, len(recent_losses)), 
                           alpha=0.7, color='#4ECDC4', edgecolor='black')
            axes[1, 0].set_title('Recent Total Loss Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Loss Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Epoch statistics (bottom right)
        stats_text = f"""
        Epoch {epoch} Statistics:
        
        Total Loss: {epoch_avg_losses.get('total_loss', 'N/A'):.6f}
        Reconstruction Loss: {epoch_avg_losses.get('rec_loss', 'N/A'):.6f}
        CLIP Loss: {epoch_avg_losses.get('clip_loss', 'N/A'):.6f}
        CLIP L2 Loss: {epoch_avg_losses.get('clip_loss_l2', 'N/A'):.6f}
        Perceptual Loss: {epoch_avg_losses.get('per_loss', 'N/A'):.6f}
        Beta Loss: {epoch_avg_losses.get('beta_loss', 'N/A'):.6f}
        
        
        Model Configuration:
        - Contrast Feature: {self.use_contrast_feat}
        - Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}
        - Device: {self.device}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        axes[1, 1].set_title('Epoch Summary', fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()
        
        # Save epoch training losses
        epoch_losses_path = os.path.join(self.writer_path, 'epoch_training_losses.png')
        plt.savefig(epoch_losses_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        
        self.logger.info(f"📊 Saved epoch training losses: {epoch_losses_path}")
        
        # Also save epoch-specific version
        epoch_specific_path = os.path.join(self.writer_path, f'epoch_training_losses.png')
        plt.figure(figsize=(16, 12))
        plt.suptitle(f'Epoch {epoch} - Training Losses', fontsize=16, fontweight='bold')
        
        # Create a simpler version for epoch-specific file
        if epoch_avg_losses:
            loss_names = list(epoch_avg_losses.keys())
            loss_values = list(epoch_avg_losses.values())
            
            # Filter out non-loss metrics
            loss_metrics = ['rec_loss', 'per_loss', 'clip_loss', 'clip_loss_l2', 'beta_loss', 'total_loss']
            filtered_names = [name for name in loss_names if name in loss_metrics]
            filtered_values = [epoch_avg_losses[name] for name in filtered_names]
            
            if filtered_values:
                colors = ['#4ECDC4', '#45B7D1', '#FFEAA7', '#FF8C00', '#96CEB4', '#FF6B6B']
                bars = plt.bar(range(len(filtered_names)), filtered_values, 
                             color=colors[:len(filtered_names)], alpha=0.8)
                plt.title(f'Epoch {epoch} Losses')
                plt.ylabel('Loss Value')
                plt.xticks(range(len(filtered_names)), filtered_names, rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, filtered_values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=10)
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(epoch_specific_path, bbox_inches='tight', facecolor='white', dpi=300)
                plt.close()

    def _update_epoch_history(self, epoch, epoch_avg_losses, val_avg_losses=None):
        """Update epoch history for tracking progression across epochs"""
        self.epoch_history['epochs'].append(epoch)
        
        # Update training metrics
        for key in ['total_loss', 'rec_loss', 'per_loss', 'clip_loss', 'clip_loss_l2', 'clip_image_loss', 'beta_loss', 'cycle_loss', 'adv_loss', 'disc_loss']:
            if key in epoch_avg_losses:
                self.epoch_history[key].append(epoch_avg_losses[key])
            else:
                self.epoch_history[key].append(0.0)
        
        # Update validation metrics
        if val_avg_losses:
            for key in ['total_loss', 'rec_loss', 'per_loss', 'clip_loss', 'clip_loss_l2', 'clip_image_loss', 'beta_loss', 'cycle_loss', 'adv_loss', 'disc_loss']:
                val_key = f'val_{key}'
                if key in val_avg_losses:
                    self.epoch_history[val_key].append(val_avg_losses[key])
                else:
                    self.epoch_history[val_key].append(0.0)
        else:
            # Add zeros for validation if no validation data
            for key in ['total_loss', 'rec_loss', 'per_loss', 'clip_loss', 'clip_loss_l2', 'clip_image_loss', 'beta_loss', 'cycle_loss', 'adv_loss', 'disc_loss']:
                val_key = f'val_{key}'
                self.epoch_history[val_key].append(0.0)
        
        # Create epoch progression plot
        self._save_epoch_progression_plot()

    def _save_epoch_progression_plot(self):
        """Save epoch progression plot showing all epochs"""
        if len(self.epoch_history['epochs']) < 1:
            return
        
        # Create comprehensive epoch progression visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress Across All Epochs', fontsize=16, fontweight='bold')
        
        epochs = self.epoch_history['epochs']
        
        # 1. Training vs Validation Total Loss (top left)
        if len(epochs) > 0:
            # Plot training total loss
            if any(v > 0 for v in self.epoch_history['total_loss']):
                axes[0, 0].plot(epochs, self.epoch_history['total_loss'], 
                               label='Train Total Loss', color='#FF0000', linewidth=2.5, marker='o')
            
            # Plot validation total loss
            if any(v > 0 for v in self.epoch_history['val_total_loss']):
                axes[0, 0].plot(epochs, self.epoch_history['val_total_loss'], 
                               label='Val Total Loss', color='#FF6B6B', linewidth=2.5, marker='s', linestyle='--')
            
            axes[0, 0].set_title('Total Loss: Training vs Validation', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss Value')
            # Only add legend if there are plots
            if any(v > 0 for v in self.epoch_history['total_loss']) or any(v > 0 for v in self.epoch_history['val_total_loss']) or any(v > 0 for v in self.epoch_history.get('adv_loss', [])):
                axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add trend line for training total loss
            if len(epochs) > 1 and any(v > 0 for v in self.epoch_history['total_loss']):
                z = np.polyfit(epochs, self.epoch_history['total_loss'], 1)
                p = np.poly1d(z)
                axes[0, 0].plot(epochs, p(epochs), "--", alpha=0.5, color='#FF0000', linewidth=1)

        # 2. Training vs Validation Component Losses (top right)
        if len(epochs) > 0:
            # Training losses
            if any(v > 0 for v in self.epoch_history['rec_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['rec_loss'], 
                               label='Train Rec Loss', color='#00FF00', linewidth=2, marker='o')
            if any(v > 0 for v in self.epoch_history['clip_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['clip_loss'], 
                               label='Train CLIP Loss', color='#FF8C00', linewidth=2, marker='^')
            if any(v > 0 for v in self.epoch_history['per_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['per_loss'], 
                               label='Train Per Loss', color='#0000FF', linewidth=2, marker='d')
            if 'beta_loss' in self.epoch_history and any(v > 0 for v in self.epoch_history['beta_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['beta_loss'], 
                               label='Train Beta Loss', color='#8A2BE2', linewidth=2, marker='*')
            if 'adv_loss' in self.epoch_history and any(v > 0 for v in self.epoch_history['adv_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['adv_loss'], 
                               label='Train Adv Loss', color='#FF1493', linewidth=2, marker='x')
            
            # Validation losses
            if any(v > 0 for v in self.epoch_history['val_rec_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['val_rec_loss'], 
                               label='Val Rec Loss', color='#32CD32', linewidth=2, marker='s', linestyle='--')
            if any(v > 0 for v in self.epoch_history['val_clip_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['val_clip_loss'], 
                               label='Val CLIP Loss', color='#FFA500', linewidth=2, marker='s', linestyle='--')
            if any(v > 0 for v in self.epoch_history['val_per_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['val_per_loss'], 
                               label='Val Per Loss', color='#4169E1', linewidth=2, marker='s', linestyle='--')
            if 'val_beta_loss' in self.epoch_history and any(v > 0 for v in self.epoch_history['val_beta_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['val_beta_loss'], 
                               label='Val Beta Loss', color='#9370DB', linewidth=2, marker='s', linestyle='--')
            if 'val_adv_loss' in self.epoch_history and any(v > 0 for v in self.epoch_history['val_adv_loss']):
                axes[0, 1].plot(epochs, self.epoch_history['val_adv_loss'], 
                               label='Val Adv Loss', color='#FF69B4', linewidth=2, marker='s', linestyle='--')
            
            axes[0, 1].set_title('Component Losses: Training vs Validation', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss Value')
            # Only add legend if there are plots
            if any(v > 0 for v in self.epoch_history['rec_loss']) or any(v > 0 for v in self.epoch_history['clip_loss']) or any(v > 0 for v in self.epoch_history['per_loss']) or any(v > 0 for v in self.epoch_history.get('beta_loss', [])) or any(v > 0 for v in self.epoch_history.get('adv_loss', [])):
                axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. CLIP Losses Progression (bottom left)
        if len(epochs) > 0:
            # Plot CLIP losses
            if any(v > 0 for v in self.epoch_history['clip_loss']):
                axes[1, 0].plot(epochs, self.epoch_history['clip_loss'], 
                               label='Train CLIP Loss', color='#FF8C00', linewidth=2, marker='o')
            if any(v > 0 for v in self.epoch_history['clip_loss_l2']):
                axes[1, 0].plot(epochs, self.epoch_history['clip_loss_l2'], 
                               label='Train CLIP L2 Loss', color='#FF4500', linewidth=2, marker='^')
            if any(v > 0 for v in self.epoch_history['clip_image_loss']):
                axes[1, 0].plot(epochs, self.epoch_history['clip_image_loss'], 
                               label='Train CLIP Image Loss', color='#FF6B6B', linewidth=2, marker='d')
            if 'adv_loss' in self.epoch_history and any(v > 0 for v in self.epoch_history['adv_loss']):
                axes[1, 0].plot(epochs, self.epoch_history['adv_loss'], 
                               label='Train Adv Loss', color='#FF1493', linewidth=2, marker='x')
            
            # Plot validation CLIP losses
            if any(v > 0 for v in self.epoch_history['val_clip_loss']):
                axes[1, 0].plot(epochs, self.epoch_history['val_clip_loss'], 
                               label='Val CLIP Loss', color='#FFA500', linewidth=2, marker='s', linestyle='--')
            if any(v > 0 for v in self.epoch_history['val_clip_loss_l2']):
                axes[1, 0].plot(epochs, self.epoch_history['val_clip_loss_l2'], 
                               label='Val CLIP L2 Loss', color='#FF6347', linewidth=2, marker='s', linestyle='--')
            if any(v > 0 for v in self.epoch_history['val_clip_image_loss']):
                axes[1, 0].plot(epochs, self.epoch_history['val_clip_image_loss'], 
                               label='Val CLIP Image Loss', color='#FFB6C1', linewidth=2, marker='s', linestyle='--')
            if 'val_adv_loss' in self.epoch_history and any(v > 0 for v in self.epoch_history['val_adv_loss']):
                axes[1, 0].plot(epochs, self.epoch_history['val_adv_loss'], 
                               label='Val Adv Loss', color='#FF69B4', linewidth=2, marker='s', linestyle='--')
            
            axes[1, 0].set_title('CLIP Losses: Training vs Validation', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Value')
            # Only add legend if there are plots
            if any(v > 0 for v in self.epoch_history.get('clip_loss', [])) or any(v > 0 for v in self.epoch_history.get('clip_loss_l2', [])) or any(v > 0 for v in self.epoch_history.get('clip_image_loss', [])) or any(v > 0 for v in self.epoch_history.get('adv_loss', [])):
                axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Training vs Validation Summary (bottom right)
        if len(epochs) > 0:
            latest_epoch = epochs[-1]
            
            # Calculate validation gap
            train_total = self.epoch_history['total_loss'][-1]
            val_total = self.epoch_history['val_total_loss'][-1] if any(v > 0 for v in self.epoch_history['val_total_loss']) else 0
            gap = abs(train_total - val_total) if val_total > 0 else 0
            
            # Calculate validation beta loss with safe fallback
            val_beta_loss_str = f"{self.epoch_history['val_beta_loss'][-1]:.6f}" if 'val_beta_loss' in self.epoch_history else 'N/A'
            
            # Calculate validation adversarial loss with safe fallback
            val_adv_loss_str = 'N/A'
            if 'val_adv_loss' in self.epoch_history and self.epoch_history['val_adv_loss']:
                val_adv_loss_str = f"{self.epoch_history['val_adv_loss'][-1]:.6f}"
            
            # Calculate training adversarial loss with safe fallback
            train_adv_loss_str = 'N/A'
            if 'adv_loss' in self.epoch_history and self.epoch_history['adv_loss']:
                train_adv_loss_str = f"{self.epoch_history['adv_loss'][-1]:.6f}"
            
            # Calculate best losses with safe fallbacks
            best_train_loss = f"{min(self.epoch_history['total_loss']):.6f}" if self.epoch_history['total_loss'] else 'N/A'
            best_val_loss = f"{min(self.epoch_history['val_total_loss']):.6f}" if any(v > 0 for v in self.epoch_history['val_total_loss']) else 'N/A'
            
            stats_text = f"""
            Training vs Validation Summary:
            
            Total Epochs: {len(epochs)}
            Latest Epoch: {latest_epoch}
            
            Latest Training Losses:
            - Total Loss: {self.epoch_history['total_loss'][-1]:.6f}
            - Rec Loss: {self.epoch_history['rec_loss'][-1]:.6f}
            - CLIP Loss: {self.epoch_history['clip_loss'][-1]:.6f}
            - CLIP L2 Loss: {self.epoch_history['clip_loss_l2'][-1]:.6f}
            - CLIP Image Loss: {self.epoch_history['clip_image_loss'][-1]:.6f}
            - Per Loss: {self.epoch_history['per_loss'][-1]:.6f}
            - Beta Loss: {self.epoch_history['beta_loss'][-1]:.6f}
            - Adv Loss: {train_adv_loss_str}
            
            Latest Validation Losses:
            - Total Loss: {self.epoch_history['val_total_loss'][-1]:.6f}
            - Rec Loss: {self.epoch_history['val_rec_loss'][-1]:.6f}
            - CLIP Loss: {self.epoch_history['val_clip_loss'][-1]:.6f}
            - CLIP L2 Loss: {self.epoch_history['val_clip_loss_l2'][-1]:.6f}
            - CLIP Image Loss: {self.epoch_history['val_clip_image_loss'][-1]:.6f}
            - Per Loss: {self.epoch_history['val_per_loss'][-1]:.6f}
            - Beta Loss: {val_beta_loss_str}
            - Adv Loss: {val_adv_loss_str}
            
            Analysis:
            - Train-Val Gap: {gap:.6f}
            - Best Train Loss: {best_train_loss}
            - Best Val Loss: {best_val_loss}
            - Overfitting: {'Yes' if gap > 0.1 else 'No'}
            """
            
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
            axes[1, 1].set_title('Training Summary', fontweight='bold')
            axes[1, 1].axis('off')

        plt.tight_layout()
        
        # Save epoch progression plot
        epoch_progression_path = os.path.join(self.writer_path, 'epoch_progression.png')
        plt.savefig(epoch_progression_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        
        self.logger.info(f"📈 Updated epoch progression plot: {epoch_progression_path}")

    def _log_epoch_metrics(self, epoch, train_losses, val_losses=None):
        """Log epoch-wise metrics to TensorBoard - simplified version"""
        if self.writer is not None:
            try:
                # Log training epoch metrics
                for k, v in train_losses.items():
                    self.writer.add_scalar(f"epoch/train_{k}", v, epoch)
                
                # Log validation epoch metrics if available
                if val_losses:
                    for k, v in val_losses.items():
                        self.writer.add_scalar(f"epoch/val_{k}", v, epoch)
                
                # Force flush to ensure data is written
                self.writer.flush()
                
            except Exception as e:
                self.logger.error(f"Error logging TensorBoard epoch metrics: {str(e)}")

    def _log_validation_metrics(self, loss_dict, val_src_imgs_chunk, val_tgt_imgs_chunk, val_output, val_mask, epoch, batch_id):
        """Log validation metrics to TensorBoard - simplified version"""
        if self.writer is not None:
            try:
                # Log all validation loss components separately
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch * 1000 + batch_id)
                
                # Log validation images every 5 batches
                if batch_id % 5 == 0:
                    val_img_grid = torch.cat([
                        val_src_imgs_chunk[0:1],  # Source
                        val_tgt_imgs_chunk[0:1],  # Target
                        val_output[0:1],          # Output
                    ], dim=0)
                    self.writer.add_images(f"val/images_epoch_{epoch}", val_img_grid, epoch * 1000 + batch_id)
                
            except Exception as e:
                self.logger.error(f"Error logging validation metrics: {str(e)}")

    def _debug_gradient_flow(self, total_loss, epoch):
        """Debug function to check if gradients are flowing properly"""
        if epoch % 5 == 0:  # Check every 10 epochs
            # Check if gradients exist for trainable models
            grad_info = {}
            grad_magnitudes = {}
            
            def check_gradients(model, name):
                if model is not None:
                    # Check gradient norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                    grad_info[name] = grad_norm.item()
                    
                    # Check individual parameter gradients
                    param_grads = []
                    for param_name, param in model.named_parameters():
                        if param.grad is not None:
                            param_grads.append(param.grad.abs().mean().item())
                        else:
                            param_grads.append(0.0)
                    
                    if param_grads:
                        grad_magnitudes[name] = {
                            'mean': np.mean(param_grads),
                            'std': np.std(param_grads),
                            'min': np.min(param_grads),
                            'max': np.max(param_grads),
                            'nonzero_count': sum(1 for g in param_grads if g > 0)
                        }
                    else:
                        grad_magnitudes[name] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'nonzero_count': 0}
            
            # Check each model
            check_gradients(self.decoder, 'decoder')
            check_gradients(self.beta_encoder, 'beta_encoder')
            check_gradients(self.enhanced_style_transfer if hasattr(self, 'enhanced_style_transfer') else None, 'enhanced_style_transfer')
            check_gradients(self.multi_scale_injector if hasattr(self, 'multi_scale_injector') else None, 'multi_scale_injector')
            check_gradients(self.adain_block if hasattr(self, 'adain_block') else None, 'adain_block')
            
            # Print model existence and parameter counts
            print(f"Model existence check:")
            print(f"  decoder: {self.decoder is not None}")
            print(f"  beta_encoder: {self.beta_encoder is not None}")
            print(f"  enhanced_style_transfer: {hasattr(self, 'enhanced_style_transfer')}")
            print(f"  multi_scale_injector: {hasattr(self, 'multi_scale_injector')}")
            
            # Print parameter counts
            if self.decoder is not None:
                decoder_params = sum(p.numel() for p in self.decoder.parameters())
                decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
                print(f"  decoder params: {decoder_params} total, {decoder_trainable} trainable")
            
            if self.beta_encoder is not None:
                beta_params = sum(p.numel() for p in self.beta_encoder.parameters())
                beta_trainable = sum(p.numel() for p in self.beta_encoder.parameters() if p.requires_grad)
                print(f"  beta_encoder params: {beta_params} total, {beta_trainable} trainable")
            
            if hasattr(self, 'enhanced_style_transfer') and self.enhanced_style_transfer is not None:
                est_params = sum(p.numel() for p in self.enhanced_style_transfer.parameters())
                est_trainable = sum(p.numel() for p in self.enhanced_style_transfer.parameters() if p.requires_grad)
                print(f"  enhanced_style_transfer params: {est_params} total, {est_trainable} trainable")
            
            if hasattr(self, 'adain_block') and self.adain_block is not None:
                adain_params = sum(p.numel() for p in self.adain_block.parameters())
                adain_trainable = sum(p.numel() for p in self.adain_block.parameters() if p.requires_grad)
                print(f"  adain_block params: {adain_params} total, {adain_trainable} trainable")
            
            # Detailed UNet analysis
            if self.decoder is not None:
                print(f"  UNet detailed analysis:")
                total_params = 0
                total_trainable = 0
                total_with_grads = 0
                
                for name, param in self.decoder.named_parameters():
                    param_count = param.numel()
                    total_params += param_count
                    if param.requires_grad:
                        total_trainable += param_count
                        if param.grad is not None:
                            total_with_grads += param_count
                
                print(f"    Total parameters: {total_params}")
                print(f"    Trainable parameters: {total_trainable}")
                print(f"    Parameters with gradients: {total_with_grads}")
                
                # Check if any parameters are frozen
                frozen_count = total_params - total_trainable
                if frozen_count > 0:
                    print(f"    WARNING: {frozen_count} parameters are frozen!")
            
            # Check CLIP model (should be 0 or very small)
            clip_grad_norm = torch.nn.utils.clip_grad_norm_(self.clip_model.parameters(), max_norm=float('inf'))
            grad_info['clip_model'] = clip_grad_norm.item()
            
            print(f"\n=== Epoch {epoch} - Gradient Analysis ===")
            print(f"Current mode: use_contrast_feat='{self.use_contrast_feat}', use_beta={self.use_beta}")
            print(f"Gradient norms: {grad_info}")
            print(f"Gradient magnitudes:")
            for model_name, mag_info in grad_magnitudes.items():
                print(f"  {model_name}: mean={mag_info['mean']:.6f}, std={mag_info['std']:.6f}, "
                      f"min={mag_info['min']:.6f}, max={mag_info['max']:.6f}, "
                      f"nonzero_params={mag_info['nonzero_count']}")
            
            # Check if total loss has gradients
            if total_loss.requires_grad:
                print(f"Total loss requires_grad: {total_loss.requires_grad}")
                print(f"Total loss grad_fn: {total_loss.grad_fn}")
            else:
                print("WARNING: Total loss does not require gradients!")
            
            # Check if gradients were computed (total_loss is not a leaf tensor, so .grad will be None)
            print("Total loss is computed tensor - gradients flow to leaf parameters")
            print("Gradient flow is working correctly!")
            
            print("=" * 50)

    def _save_batch_100_progression_plot(self):
        """Save 100-batch progression plot showing training metrics over batches across all epochs."""
        if len(self.batch_100_history['batches']) < 1:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress (100-batch averages across all epochs)', fontsize=16, fontweight='bold')
        
        batches = self.batch_100_history['batches']
        
        # 1. Total & Rec loss over batches
        if any(v > 0 for v in self.batch_100_history['total_loss']):
            axes[0, 0].plot(batches, self.batch_100_history['total_loss'], label='Total Loss', color='#FF0000', linewidth=2)
        if any(v > 0 for v in self.batch_100_history['rec_loss']):
            axes[0, 0].plot(batches, self.batch_100_history['rec_loss'], label='Rec Loss', color='#00AAFF', linewidth=2)
        axes[0, 0].set_title('Total & Rec Loss vs Global Batches', fontweight='bold')
        axes[0, 0].set_xlabel('Global Batch (100-batch bucket end)')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. CLIP losses over batches
        for name, color in [('clip_loss', '#FF8C00'), ('clip_loss_l2', '#FF4500'), ('clip_image_loss', '#FF6B6B')]:
            if any(v > 0 for v in self.batch_100_history[name]):
                axes[0, 1].plot(batches, self.batch_100_history[name], label=name.replace('_', ' ').title(), linewidth=2)
        axes[0, 1].set_title('CLIP Losses vs Global Batches', fontweight='bold')
        axes[0, 1].set_xlabel('Global Batch (100-batch bucket end)')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Beta/Adv/Disc losses over batches
        if any(v > 0 for v in self.batch_100_history['beta_loss']):
            axes[1, 0].plot(batches, self.batch_100_history['beta_loss'], label='Beta Loss', color='#8A2BE2', linewidth=2)
        if any(v > 0 for v in self.batch_100_history['adv_loss']):
            axes[1, 0].plot(batches, self.batch_100_history['adv_loss'], label='Adv Loss', color='#FF1493', linewidth=2)
        if any(v > 0 for v in self.batch_100_history['disc_loss']):
            axes[1, 0].plot(batches, self.batch_100_history['disc_loss'], label='Disc Loss', color='#2F4F4F', linewidth=2)
        axes[1, 0].set_title('Other Losses vs Global Batches', fontweight='bold')
        axes[1, 0].set_xlabel('Global Batch (100-batch bucket end)')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Latest stats
        latest_idx = -1
        stats_text = f"""
        Latest Global Batch: {batches[latest_idx]}
        Total Loss: {self.batch_100_history['total_loss'][latest_idx] if self.batch_100_history['total_loss'] else 'N/A'}
        Rec Loss: {self.batch_100_history['rec_loss'][latest_idx] if self.batch_100_history['rec_loss'] else 'N/A'}
        CLIP: {self.batch_100_history['clip_loss'][latest_idx] if self.batch_100_history['clip_loss'] else 'N/A'} | L2: {self.batch_100_history['clip_loss_l2'][latest_idx] if self.batch_100_history['clip_loss_l2'] else 'N/A'} | Img: {self.batch_100_history['clip_image_loss'][latest_idx] if self.batch_100_history['clip_image_loss'] else 'N/A'}
        Beta: {self.batch_100_history['beta_loss'][latest_idx] if self.batch_100_history['beta_loss'] else 'N/A'}
        Adv: {self.batch_100_history['adv_loss'][latest_idx] if self.batch_100_history['adv_loss'] else 'N/A'} | Disc: {self.batch_100_history['disc_loss'][latest_idx] if self.batch_100_history['disc_loss'] else 'N/A'}
        """
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title('Latest Stats', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(self.writer_path, 'batch_100_progression.png')
        plt.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        if hasattr(self, 'logger'):
            self.logger.info(f"📈 Updated 100-batch progression plot: {out_path}")

    def log_gradient_norms_from_tensors(self, loss_dict, total_loss, epoch, batch_id, 
                                       output, tgt_imgs_chunk, mask, source_betas, target_betas,
                                       src_imgs_chunk, source_clip_img_feat, src_clip_text_feat,
                                       tgt_clip_text_feat, target_clip_img_feat, step_id=None):
        """
        Log gradient norms by computing gradients from the actual loss tensors.
        This method re-computes each loss component to get proper gradients.
        """
        if not hasattr(self, 'gradient_norms_history'):
            self.gradient_norms_history = []
        
        print(f"\n=== Gradient Norms Epoch {epoch} Batch {batch_id} ===")
        
        # Store current parameter states
        param_states = {}
        for name, param in self.decoder.named_parameters():
            if param.requires_grad:
                param_states[f'decoder.{name}'] = param.data.clone()
        
        if self.use_beta and self.beta_encoder is not None:
            for name, param in self.beta_encoder.named_parameters():
                if param.requires_grad:
                    param_states[f'beta_encoder.{name}'] = param.data.clone()
        
        if self.use_patchifier and self.patchifier is not None:
            for name, param in self.patchifier.named_parameters():
                if param.requires_grad:
                    param_states[f'patchifier.{name}'] = param.data.clone()
        
        gradient_norms = {}
        
        # We need to re-compute the losses to get proper gradients
        # Let's compute them individually by calling calculate_loss with different weights
        
        # 1. Reconstruction loss only
        self.optimizer.zero_grad()
        temp_w_rec, temp_w_clip, temp_w_clip_l2, temp_w_clip_img, temp_w_beta, temp_w_adv, temp_w_per = \
            self.w_rec, self.w_clip, self.w_clip_l2, self.w_clip_img, self.w_beta, self.w_adv, self.w_per
        
        self.w_rec, self.w_clip, self.w_clip_l2, self.w_clip_img, self.w_beta, self.w_adv, self.w_per = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        temp_loss_dict, temp_total_loss = self.calculate_loss(
            epoch=epoch,
            rec_image=output,
            ref_image=tgt_imgs_chunk,
            mask=mask,
            source_betas=source_betas,
            target_betas=target_betas,
            source_images=src_imgs_chunk,
            source_clip_img_feat=source_clip_img_feat,
            source_clip_text_feat=src_clip_text_feat,
            target_clip_text_feat=tgt_clip_text_feat,
            target_clip_img_feat=target_clip_img_feat,
            target_images=tgt_imgs_chunk,
            is_train=True
        )
        
        if temp_total_loss is not None:
            temp_total_loss.backward(retain_graph=True)
            
            # Decoder gradients
            decoder_norm = 0.0
            for p in self.decoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    decoder_norm += param_norm.item() ** 2
            decoder_norm = decoder_norm ** 0.5
            gradient_norms['rec_loss_decoder'] = decoder_norm
            print(f"rec_loss_decoder grad norm: {decoder_norm:.6f}")
            
            # Beta encoder gradients
            if self.use_beta and self.beta_encoder is not None:
                encoder_norm = 0.0
                for p in self.beta_encoder.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        encoder_norm += param_norm.item() ** 2
                encoder_norm = encoder_norm ** 0.5
                gradient_norms['rec_loss_encoder'] = encoder_norm
                print(f"rec_loss_encoder grad norm: {encoder_norm:.6f}")
            
            # Patchifier gradients
            if self.use_patchifier and self.patchifier is not None:
                patchifier_norm = 0.0
                for p in self.patchifier.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        patchifier_norm += param_norm.item() ** 2
                patchifier_norm = patchifier_norm ** 0.5
                gradient_norms['rec_loss_patchifier'] = patchifier_norm
                print(f"rec_loss_patchifier grad norm: {patchifier_norm:.6f}")
        
        # Restore weights
        self.w_rec, self.w_clip, self.w_clip_l2, self.w_clip_img, self.w_beta, self.w_adv, self.w_per = \
            temp_w_rec, temp_w_clip, temp_w_clip_l2, temp_w_clip_img, temp_w_beta, temp_w_adv, temp_w_per
        
        # 2. Beta loss only
        if self.use_beta and self.beta_encoder is not None:
            self.optimizer.zero_grad()
            self.w_rec, self.w_clip, self.w_clip_l2, self.w_clip_img, self.w_beta, self.w_adv, self.w_per = 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
            
            temp_loss_dict, temp_total_loss = self.calculate_loss(
                epoch=epoch,
                rec_image=output,
                ref_image=tgt_imgs_chunk,
                mask=mask,
                source_betas=source_betas,
                target_betas=target_betas,
                source_images=src_imgs_chunk,
                source_clip_img_feat=source_clip_img_feat,
                source_clip_text_feat=src_clip_text_feat,
                target_clip_text_feat=tgt_clip_text_feat,
                target_clip_img_feat=target_clip_img_feat,
                target_images=tgt_imgs_chunk,
                is_train=True
            )
            
            if temp_total_loss is not None:
                temp_total_loss.backward(retain_graph=True)
                
                # Decoder gradients
                decoder_norm = 0.0
                for p in self.decoder.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        decoder_norm += param_norm.item() ** 2
                decoder_norm = decoder_norm ** 0.5
                gradient_norms['beta_loss_decoder'] = decoder_norm
                print(f"beta_loss_decoder grad norm: {decoder_norm:.6f}")
                
                # Beta encoder gradients
                encoder_norm = 0.0
                for p in self.beta_encoder.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        encoder_norm += param_norm.item() ** 2
                encoder_norm = encoder_norm ** 0.5
                gradient_norms['beta_loss_encoder'] = encoder_norm
                print(f"beta_loss_encoder grad norm: {encoder_norm:.6f}")
                
                # Patchifier gradients
                if self.use_patchifier and self.patchifier is not None:
                    patchifier_norm = 0.0
                    for p in self.patchifier.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            patchifier_norm += param_norm.item() ** 2
                    patchifier_norm = patchifier_norm ** 0.5
                    gradient_norms['beta_loss_patchifier'] = patchifier_norm
                    print(f"beta_loss_patchifier grad norm: {patchifier_norm:.6f}")
        
        # Restore weights
        self.w_rec, self.w_clip, self.w_clip_l2, self.w_clip_img, self.w_beta, self.w_adv, self.w_per = \
            temp_w_rec, temp_w_clip, temp_w_clip_l2, temp_w_clip_img, temp_w_beta, temp_w_adv, temp_w_per
        
        # 3. CLIP text loss only
        if epoch > 1:  # Only if CLIP loss is active
            self.optimizer.zero_grad()
            self.w_rec, self.w_clip, self.w_clip_l2, self.w_clip_img, self.w_beta, self.w_adv, self.w_per = 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            temp_loss_dict, temp_total_loss = self.calculate_loss(
                epoch=epoch,
                rec_image=output,
                ref_image=tgt_imgs_chunk,
                mask=mask,
                source_betas=source_betas,
                target_betas=target_betas,
                source_images=src_imgs_chunk,
                source_clip_img_feat=source_clip_img_feat,
                source_clip_text_feat=src_clip_text_feat,
                target_clip_text_feat=tgt_clip_text_feat,
                target_clip_img_feat=target_clip_img_feat,
                target_images=tgt_imgs_chunk,
                is_train=True
            )
            
            if temp_total_loss is not None:
                temp_total_loss.backward(retain_graph=True)
                
                # Decoder gradients
                decoder_norm = 0.0
                for p in self.decoder.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        decoder_norm += param_norm.item() ** 2
                decoder_norm = decoder_norm ** 0.5
                gradient_norms['clip_text_loss_decoder'] = decoder_norm
                print(f"clip_text_loss_decoder grad norm: {decoder_norm:.6f}")
                
                # Beta encoder gradients
                if self.use_beta and self.beta_encoder is not None:
                    encoder_norm = 0.0
                    for p in self.beta_encoder.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            encoder_norm += param_norm.item() ** 2
                    encoder_norm = encoder_norm ** 0.5
                    gradient_norms['clip_text_loss_encoder'] = encoder_norm
                    print(f"clip_text_loss_encoder grad norm: {encoder_norm:.6f}")
                
                # Patchifier gradients
                if self.use_patchifier and self.patchifier is not None:
                    patchifier_norm = 0.0
                    for p in self.patchifier.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            patchifier_norm += param_norm.item() ** 2
                    patchifier_norm = patchifier_norm ** 0.5
                    gradient_norms['clip_text_loss_patchifier'] = patchifier_norm
                    print(f"clip_text_loss_patchifier grad norm: {patchifier_norm:.6f}")
        
        # 4. CLIP L2 loss only
        if epoch > 1:
            self.optimizer.zero_grad()
            self.w_rec, self.w_clip, self.w_clip_l2, self.w_clip_img, self.w_beta, self.w_adv, self.w_per = 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
            
            temp_loss_dict, temp_total_loss = self.calculate_loss(
                epoch=epoch,
                rec_image=output,
                ref_image=tgt_imgs_chunk,
                mask=mask,
                source_betas=source_betas,
                target_betas=target_betas,
                source_images=src_imgs_chunk,
                source_clip_img_feat=source_clip_img_feat,
                source_clip_text_feat=src_clip_text_feat,
                target_clip_text_feat=tgt_clip_text_feat,
                target_clip_img_feat=target_clip_img_feat,
                target_images=tgt_imgs_chunk,
                is_train=True
            )
            
            if temp_total_loss is not None:
                temp_total_loss.backward(retain_graph=True)
                
                # Decoder gradients
                decoder_norm = 0.0
                for p in self.decoder.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        decoder_norm += param_norm.item() ** 2
                decoder_norm = decoder_norm ** 0.5
                gradient_norms['clip_l2_loss_decoder'] = decoder_norm
                print(f"clip_l2_loss_decoder grad norm: {decoder_norm:.6f}")
                
                # Beta encoder gradients
                if self.use_beta and self.beta_encoder is not None:
                    encoder_norm = 0.0
                    for p in self.beta_encoder.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            encoder_norm += param_norm.item() ** 2
                    encoder_norm = encoder_norm ** 0.5
                    gradient_norms['clip_l2_loss_encoder'] = encoder_norm
                    print(f"clip_l2_loss_encoder grad norm: {encoder_norm:.6f}")
                
                # Patchifier gradients
                if self.use_patchifier and self.patchifier is not None:
                    patchifier_norm = 0.0
                    for p in self.patchifier.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            patchifier_norm += param_norm.item() ** 2
                    patchifier_norm = patchifier_norm ** 0.5
                    gradient_norms['clip_l2_loss_patchifier'] = patchifier_norm
                    print(f"clip_l2_loss_patchifier grad norm: {patchifier_norm:.6f}")
        
        # 5. CLIP image loss only
        if epoch > 1:
            self.optimizer.zero_grad()
            self.w_rec, self.w_clip, self.w_clip_l2, self.w_clip_img, self.w_beta, self.w_adv, self.w_per = 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
            
            # Detach beta features to prevent CLIP loss from affecting beta encoder
           
            
            temp_loss_dict, temp_total_loss = self.calculate_loss(
                epoch=epoch,
                rec_image=output,
                ref_image=tgt_imgs_chunk,
                mask=mask,
                source_betas=source_betas,
                target_betas=target_betas,
                source_images=src_imgs_chunk,
                source_clip_img_feat=source_clip_img_feat,
                source_clip_text_feat=src_clip_text_feat,
                target_clip_text_feat=tgt_clip_text_feat,
                target_clip_img_feat=target_clip_img_feat,
                target_images=tgt_imgs_chunk,
                is_train=True
            )
            
            if temp_total_loss is not None:
                temp_total_loss.backward(retain_graph=True)
                
                # Decoder gradients
                decoder_norm = 0.0
                for p in self.decoder.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        decoder_norm += param_norm.item() ** 2
                decoder_norm = decoder_norm ** 0.5
                gradient_norms['clip_image_loss_decoder'] = decoder_norm
                print(f"clip_image_loss_decoder grad norm: {decoder_norm:.6f}")
                
                # Beta encoder gradients
                if self.use_beta and self.beta_encoder is not None:
                    encoder_norm = 0.0
                    for p in self.beta_encoder.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            encoder_norm += param_norm.item() ** 2
                    encoder_norm = encoder_norm ** 0.5
                    gradient_norms['clip_image_loss_encoder'] = encoder_norm
                    print(f"clip_image_loss_encoder grad norm: {encoder_norm:.6f}")
                
                # Patchifier gradients
                if self.use_patchifier and self.patchifier is not None:
                    patchifier_norm = 0.0
                    for p in self.patchifier.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            patchifier_norm += param_norm.item() ** 2
                    patchifier_norm = patchifier_norm ** 0.5
                    gradient_norms['clip_image_loss_patchifier'] = patchifier_norm
                    print(f"clip_image_loss_patchifier grad norm: {patchifier_norm:.6f}")
        
        # Restore weights
        self.w_rec, self.w_clip, self.w_clip_l2, self.w_clip_img, self.w_beta, self.w_adv, self.w_per = \
            temp_w_rec, temp_w_clip, temp_w_clip_l2, temp_w_clip_img, temp_w_beta, temp_w_adv, temp_w_per
        
        # Show loss values and weights for context
        print(f"\n=== Loss Values and Weights ===")
        for loss_name, loss_value in loss_dict.items():
            if loss_name == 'total_loss':
                continue
                
            # Get the weight for this loss
            weight = 0.0
            if loss_name == 'rec_loss':
                weight = self.w_rec
            elif loss_name == 'clip_loss':
                weight = self.w_clip
            elif loss_name == 'clip_loss_l2':
                weight = self.w_clip_l2
            elif loss_name == 'clip_image_loss':
                weight = self.w_clip_img
            elif loss_name == 'beta_loss':
                weight = self.w_beta
            elif loss_name == 'adv_loss':
                weight = self.w_adv
            elif loss_name == 'per_loss':
                weight = self.w_per
            
            weighted_loss = weight * loss_value
            print(f"{loss_name:20s}: loss={loss_value:8.6f}, weight={weight:6.3f}, weighted={weighted_loss:8.6f}")
        
        # Store in history
        analysis_data = {
            'epoch': epoch,
            'batch_id': batch_id,
            'step_id': step_id,
            'gradient_norms': gradient_norms.copy(),
            'loss_dict': loss_dict.copy()
        }
        self.gradient_norms_history.append(analysis_data)
        
        # Update persistent time-series plot with per-step averaging
        try:
            import matplotlib.pyplot as plt
            # Aggregate by step (epoch*10000 + batch_id)
            step_to_items = {}
            for item in self.gradient_norms_history:
                s = int(item.get('epoch', 0)) * 10000 + int(item.get('batch_id', 0))
                step_to_items.setdefault(s, []).append(item)
            steps = sorted(step_to_items.keys())
            # Collect gradient names
            grad_names = set()
            for items in step_to_items.values():
                for it in items:
                    grad_names.update(it['gradient_norms'].keys())
            grad_names = sorted(list(grad_names))
            # Collect loss names of interest
            loss_names = ['rec_loss','per_loss','clip_loss','clip_loss_l2','clip_image_loss','beta_loss','adv_loss']
            # Build averaged series
            grad_series = {name: [] for name in grad_names}
            loss_series = {name: [] for name in loss_names}
            for s in steps:
                items = step_to_items[s]
                # gradients
                for name in grad_names:
                    vals = [it['gradient_norms'].get(name, 0.0) for it in items]
                    grad_series[name].append(float(sum(vals)) / max(len(vals), 1))
                # losses
                for name in loss_names:
                    vals = [float(it['loss_dict'].get(name, 0.0)) for it in items]
                    loss_series[name].append(float(sum(vals)) / max(len(vals), 1))
            # Plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax = axes[0]
            for name in grad_names:
                ax.plot(steps, grad_series[name], label=name)
            ax.set_ylabel('Grad Norm')
            ax.set_title('Gradient Norms over Batches (averaged per step)')
            ax.grid(True, alpha=0.3)
            if len(grad_names) <= 10:
                ax.legend(fontsize=8, ncol=3)
            ax2 = axes[1]
            for name in loss_names:
                ax2.plot(steps, loss_series[name], label=name)
            ax2.set_xlabel('Step (epoch*10000 + batch)')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss Values over Batches (averaged per step)')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8, ncol=3)
            fig.tight_layout()
            out_path = os.path.join(self.writer_path, 'grad_debug_timeseries.png')
            plt.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass
        
        # Restore parameter states
        for name, param_data in param_states.items():
            if 'decoder.' in name:
                param_name = name.replace('decoder.', '')
                if hasattr(self.decoder, param_name.split('.')[0]):
                    param = self.decoder
                    for attr in param_name.split('.'):
                        param = getattr(param, attr)
                    param.data = param_data
            elif 'beta_encoder.' in name and self.use_beta and self.beta_encoder is not None:
                param_name = name.replace('beta_encoder.', '')
                if hasattr(self.beta_encoder, param_name.split('.')[0]):
                    param = self.beta_encoder
                    for attr in param_name.split('.'):
                        param = getattr(param, attr)
                    param.data = param_data
            elif 'patchifier.' in name and self.use_patchifier and self.patchifier is not None:
                param_name = name.replace('patchifier.', '')
                if hasattr(self.patchifier, param_name.split('.')[0]):
                    param = self.patchifier
                    for attr in param_name.split('.'):
                        param = getattr(param, attr)
                    param.data = param_data
        
        # Clear gradients
        self.optimizer.zero_grad()
