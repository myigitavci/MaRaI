import numbers
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop, ColorJitter, Grayscale,RandomRotation, GaussianBlur, RandomHorizontalFlip, RandomAffine
import numpy as np
import scipy.ndimage

# from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
# from .utils import to_2tuple

# Define constants inline
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class VolumeNormalizationError(Exception):
    """Exception raised when volume normalization fails and should be skipped."""
    pass


@dataclass
class PreprocessCfg:
    size: Union[int, Tuple[int, int]] = 224
    mode: str = 'RGB'
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'
    fill_color: int = 0

    def __post_init__(self):
        assert self.mode in ('RGB',)

    @property
    def num_channels(self):
        return 3

    @property
    def input_size(self):
        return (self.num_channels,) + to_2tuple(self.size)

_PREPROCESS_KEYS = set(asdict(PreprocessCfg()).keys())


def merge_preprocess_dict(
        base: Union[PreprocessCfg, Dict],
        overlay: Dict,
):
    """ Merge overlay key-value pairs on top of base preprocess cfg or dict.
    Input dicts are filtered based on PreprocessCfg fields.
    """
    if isinstance(base, PreprocessCfg):
        base_clean = asdict(base)
    else:
        base_clean = {k: v for k, v in base.items() if k in _PREPROCESS_KEYS}
    if overlay:
        overlay_clean = {k: v for k, v in overlay.items() if k in _PREPROCESS_KEYS and v is not None}
        base_clean.update(overlay_clean)
    return base_clean


def merge_preprocess_kwargs(base: PreprocessCfg, **kwargs):
    return merge_preprocess_dict(base, kwargs)


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False

    # params for simclr_jitter_gray
    color_jitter_prob: float = None
    gray_scale_prob: float = None


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class ResizeKeepRatio:
    """ Resize and Keep Ratio

    Copy & paste from `timm`
    """

    def __init__(
            self,
            size,
            longest=0.,
            interpolation=InterpolationMode.BICUBIC,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)  # [0, 1] where 0 == shortest edge, 1 == longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
            img,
            target_size,
            longest,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        """Get parameters
        """
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(random_aspect_range[0], random_aspect_range[1])
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)
        size = [round(x * f / ratio) for x, f in zip(source_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img, self.size, self.longest,
            self.random_scale_prob, self.random_scale_range,
            self.random_aspect_prob, self.random_aspect_range
        )
        img = F.resize(img, size, self.interpolation)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += f', interpolation={self.interpolation})'
        format_string += f', longest={self.longest:.3f})'
        return format_string


def center_crop_or_pad(img: torch.Tensor, output_size: List[int], fill=0) -> torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size, fill=0):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def _convert_to_rgb(image):
    return image.convert('RGB')


class color_jitter(object):
    """
    Apply Color Jitter to the PIL image with a specified probability.
    """
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., p=0.8):
        assert 0. <= p <= 1.
        self.p = p
        self.transf = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Gray Scale to the PIL image with a specified probability.
    """
    def __init__(self, p=0.2):
        assert 0. <= p <= 1.
        self.p = p
        self.transf = Grayscale(num_output_channels=3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def image_transform(
        image_size: Union[int, Tuple[int, int]],
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_mode: Optional[str] = None,
        interpolation: Optional[str] = None,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        rgb: bool = False,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    interpolation = interpolation or 'bicubic'
    assert interpolation in ['bicubic', 'bilinear', 'random']
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for inference if set
    interpolation_mode = InterpolationMode.BILINEAR if interpolation == 'bilinear' else InterpolationMode.BICUBIC

    resize_mode = resize_mode or 'shortest'
    assert resize_mode in ('shortest', 'longest', 'squash')

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()

    if rgb:
        normalize = Normalize(mean=mean, std=std)
        if is_train:
            aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
            use_timm = aug_cfg_dict.pop('use_timm', False)
            if use_timm:
                from timm.data import create_transform
                if isinstance(image_size, (tuple, list)):
                    assert len(image_size) >= 2
                    input_size = (3,) + image_size[-2:]
                else:
                    input_size = (3, image_size, image_size)
                aug_cfg_dict.setdefault('color_jitter', None)
                aug_cfg_dict.pop('color_jitter_prob', None)
                aug_cfg_dict.pop('gray_scale_prob', None)
                train_transform = create_transform(
                    input_size=input_size,
                    is_training=True,
                    hflip=0.,
                    mean=mean,
                    std=std,
                    re_mode='pixel',
                    interpolation=interpolation,
                    **aug_cfg_dict,
                )
            else:
                train_transform = [
                    RandomResizedCrop(
                        image_size,
                        scale=aug_cfg_dict.pop('scale'),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    _convert_to_rgb,
                ]
                train_transform.extend([RandomAffine(degrees=(-20, 20), 
                                                     translate=(0.3, 0.3),
                                                     scale=(0.8, 1.2))])
                train_transform.extend([GaussianBlur(kernel_size=3)])
                train_transform.extend([RandomHorizontalFlip()])
                if aug_cfg.color_jitter_prob:
                    assert aug_cfg.color_jitter is not None and len(aug_cfg.color_jitter) == 4
                    train_transform.extend([
                        color_jitter(*aug_cfg.color_jitter, p=aug_cfg.color_jitter_prob)
                    ])
                if aug_cfg.gray_scale_prob:
                    train_transform.extend([
                        gray_scale(aug_cfg.gray_scale_prob)
                    ])
                train_transform.extend([
                    ToTensor(),
                    normalize,
                ])
                train_transform = Compose(train_transform)
                if aug_cfg_dict:
                    warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
            return train_transform
        else:
            if resize_mode == 'longest':
                transforms = [
                    ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
                    CenterCropOrPad(image_size, fill=fill_color)
                ]
            elif resize_mode == 'squash':
                if isinstance(image_size, int):
                    image_size = (image_size, image_size)
                transforms = [
                    Resize(image_size, interpolation=interpolation_mode),
                ]
            else:
                assert resize_mode == 'shortest'
                if not isinstance(image_size, (tuple, list)):
                    image_size = (image_size, image_size)
                if image_size[0] == image_size[1]:
                    transforms = [
                        Resize(image_size[0], interpolation=interpolation_mode)
                    ]
                else:
                    transforms = [ResizeKeepRatio(image_size)]
                transforms += [CenterCrop(image_size)]
            transforms.extend([
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
            return Compose(transforms)
    else:
        # Grayscale: do not convert to RGB, use classic normalization (scale to [0,1])
        # def classic_normalize(img):
        #     # img is a torch tensor
        #     if isinstance(img, torch.Tensor):
        #         min_val = img.min()
        #         max_val = img.max()
        #         if max_val > min_val:
        #             return (img - min_val) / (max_val - min_val)
        #         else:
        #             return img - min_val  # fallback if all values are the same
        #     else:
        #         return img
        # def classic_normalize(img):
        #     # img is a torch tensor
        #     if isinstance(img, torch.Tensor):
        #         print(f"[Norm98] min: {img.min():.6f}, max: {img.max():.6f}, mean: {img.mean():.6f}, std: {img.std():.6f}")
        #         return img/255  # fallback if all values are the same
            
        #     else:
        #         return img/255
        if is_train:
            train_transform = [
                RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
                # No _convert_to_rgb
                RandomAffine(degrees=(-20, 20), translate=(0.3, 0.3), scale=(0.8, 1.2)),
                GaussianBlur(kernel_size=3),
                RandomHorizontalFlip(),
                ToTensor(),
                # classic_normalize,
            ]
            return Compose(train_transform)
        else:
            if resize_mode == 'longest':
                transforms = [
                    ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
                    CenterCropOrPad(image_size, fill=fill_color)
                ]
            elif resize_mode == 'squash':
                if isinstance(image_size, int):
                    image_size = (image_size, image_size)
                transforms = [
                    Resize(image_size, interpolation=interpolation_mode),
                ]
            else:
                assert resize_mode == 'shortest'
                if not isinstance(image_size, (tuple, list)):
                    image_size = (image_size, image_size)
                if image_size[0] == image_size[1]:
                    transforms = [
                        Resize(image_size[0], interpolation=interpolation_mode)
                    ]
                else:
                    transforms = [ResizeKeepRatio(image_size)]
                transforms += [CenterCrop(image_size)]
            transforms.extend([
                # No _convert_to_rgb
                ToTensor(),
                # classic_normalize,
            ])
            return Compose(transforms)


def image_transform_v2(
        cfg: PreprocessCfg,
        is_train: bool,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        vis_3d: bool = False,
        rgb: bool = False,
):
    """
    Unified transform function for 2D and 3D. If vis_3d is True, applies 3D transforms, else 2D.
    """
    if vis_3d:
        return volume_transform_v2(cfg, is_train, aug_cfg)
    # --- original 2D logic below ---
    return image_transform(
        image_size=cfg.size,
        is_train=is_train,
        mean=cfg.mean,
        std=cfg.std,
        interpolation=cfg.interpolation,
        resize_mode=cfg.resize_mode,
        fill_color=cfg.fill_color,
        aug_cfg=aug_cfg,
        rgb=rgb,  # use RGB images instead of grayscale
    )


class Volume3DTransform:
    """
    Apply a sequence of 3D transforms to a 3D volume (C, D, H, W) or (D, H, W).
    Supports fixed cropping (for registered images), resizing, random affine, and normalization.
    Uses PyTorch operations for better performance.
    """
    def __init__(self, image_size=128, is_train=True, mean=0.0, std=1.0, p_flip=0.5, affine_degrees=15, affine_translate=0.1, affine_scale=(0.9, 1.1), crop_coords={'d_min': 22, 'd_max': 170, 'h_min': 38, 'h_max': 213, 'w_min': 86, 'w_max': 237}, save_examples=False, example_dir='crop_examples', clip_image_size=None, with_skull=False):
        """
        Args:
            image_size: Target size after resizing (default: 128)
            is_train: Training mode (enables augmentation)
            crop_coords: Dict with keys d_min, d_max, h_min, h_max, w_min, w_max
                         Default: crop coordinates found for registered OASIS dataset
                         Set to None to disable cropping
            save_examples: If True, saves example volumes after cropping (for verification)
            example_dir: Directory to save example volumes
            clip_image_size: Optional DIFFERENT target size for CLIP encoder input. 
                             If provided, __call__ returns a dict {'image': ..., 'clip_image': ...}
            with_skull: If True, use crop coordinates that include skull; if False, use no-skull coordinates
        """
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size, image_size)
        self.clip_image_size = None
        if clip_image_size is not None:
             self.clip_image_size = clip_image_size if isinstance(clip_image_size, (tuple, list)) else (clip_image_size, clip_image_size, clip_image_size)
             
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.p_flip = p_flip
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.crop_coords = crop_coords
        self.with_skull = with_skull
        self.save_examples = save_examples
        self.example_dir = example_dir
        self._examples_saved = 0
        self._max_examples = 5  # Save only first 5 examples

    def crop_fixed(self, volume):
        """
        Apply fixed crop coordinates (for registered images).
        
        Args:
            volume: (C, D, H, W) or (D, H, W) torch tensor
        
        Returns:
            Cropped volume torch tensor
        """
        # Choose crop coordinates based on with_skull parameter
        if self.with_skull:
            # with skull
            crop_coords = {'d_min': 12, 'd_max': 180, 'h_min': 20, 'h_max': 227, 'w_min': 82, 'w_max': 245}
        else:
            # no skull
            crop_coords = {'d_min': 22, 'd_max': 170, 'h_min': 28, 'h_max': 214, 'w_min': 90, 'w_max': 237}

        is_4d = len(volume.shape) == 4
        
        d_min = crop_coords['d_min']
        d_max = crop_coords['d_max']
        h_min = crop_coords['h_min']
        h_max = crop_coords['h_max']
        w_min = crop_coords['w_min']
        w_max = crop_coords['w_max']
        
        # Crop
        if is_4d:
            cropped = volume[:, d_min:d_max, h_min:h_max, w_min:w_max]
        else:
            cropped = volume[d_min:d_max, h_min:h_max, w_min:w_max]
        
        return cropped
    
    def save_crop_example(self, original_volume, cropped_volume, index):
        """
        Save example volumes before and after cropping for verification.
        """
        import os
        import nibabel as nib
        
        os.makedirs(self.example_dir, exist_ok=True)
        
        # Helper to convert to numpy 3D
        def to_numpy_3d(vol):
            if isinstance(vol, torch.Tensor):
                vol = vol.detach().cpu().numpy()
            if vol.ndim == 4:
                return vol[0] if vol.shape[0] == 1 else vol.mean(axis=0)
            return vol
        
        orig_3d = to_numpy_3d(original_volume)
        crop_3d = to_numpy_3d(cropped_volume)
        
        # Save as NIfTI
        orig_nii = nib.Nifti1Image(orig_3d, affine=np.eye(4))
        crop_nii = nib.Nifti1Image(crop_3d, affine=np.eye(4))
        
        orig_path = os.path.join(self.example_dir, f'example_{index:03d}_original.nii.gz')
        crop_path = os.path.join(self.example_dir, f'example_{index:03d}_cropped.nii.gz')
        
        nib.save(orig_nii, orig_path)
        nib.save(crop_nii, crop_path)
        
        print(f"💾 Saved crop example {index}:")
        print(f"   Original: {orig_3d.shape} → {orig_path}")
        print(f"   Cropped:  {crop_3d.shape} → {crop_path}")
    
    def resize(self, volume, target_size=None):
        # volume: (C, D, H, W) or (D, H, W) torch tensor
        if not isinstance(volume, torch.Tensor):
             volume = torch.from_numpy(volume).float()

        # Check for NaN in input
        if torch.isnan(volume).any():
            raise ValueError("Input volume contains NaN values")
            
        orig_dim = volume.dim()
        if orig_dim == 3:
            # Add channel and batch dimension: (1, 1, D, H, W) for interpolate
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 4:
            # Add batch dimension: (1, C, D, H, W)
            volume = volume.unsqueeze(0)
        else:
             raise ValueError("Volume must be 3D or 4D")

        size_to_use = target_size if target_size is not None else self.image_size
        
        # PyTorch interpolate expects (Batch, Channel, D, H, W)
        resized = F.interpolate(volume, size=size_to_use, mode='trilinear', align_corners=False)

        # Remove extra dimensions
        if orig_dim == 3:
            resized = resized.squeeze(0).squeeze(0)
        elif orig_dim == 4:
            resized = resized.squeeze(0)
            
        # Check for NaN after resize
        if torch.isnan(resized).any():
            raise ValueError("NaN detected after resize operation")
            
        return resized

    def random_horizontal_flip(self, volume):
        # Flip along W axis (last dim)
        if random.random() < self.p_flip:
            # torch.flip is widely supported
            return torch.flip(volume, dims=[-1])
        return volume

    def random_affine(self, volume):
        # Only apply for training
        if not self.is_train:
            return volume
            
        # Check for NaN before affine
        if torch.isnan(volume).any():
            raise ValueError("NaN detected before affine transform")

        # Create affine matrix manually
        angle_x = random.uniform(-self.affine_degrees, self.affine_degrees)
        angle_y = random.uniform(-self.affine_degrees, self.affine_degrees)
        angle_z = random.uniform(-self.affine_degrees, self.affine_degrees)
        
        ax, ay, az = np.deg2rad([angle_x, angle_y, angle_z])

        # Rotation matrices
        Rx = torch.tensor([[1, 0, 0],
                           [0, np.cos(ax), -np.sin(ax)],
                           [0, np.sin(ax), np.cos(ax)]], dtype=torch.float32)
        Ry = torch.tensor([[np.cos(ay), 0, np.sin(ay)],
                           [0, 1, 0],
                           [-np.sin(ay), 0, np.cos(ay)]], dtype=torch.float32)
        Rz = torch.tensor([[np.cos(az), -np.sin(az), 0],
                           [np.sin(az), np.cos(az), 0],
                           [0, 0, 1]], dtype=torch.float32)

        # Combined rotation: Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx

        # Scale
        scale = random.uniform(self.affine_scale[0], self.affine_scale[1])
        S = torch.diag(torch.tensor([scale, scale, scale], dtype=torch.float32))

        # Combined affine (rotation + scale)
        # Note: In grid_sample, the transformation maps output coordinates to input coordinates.
        # So we typically use the inverse. However, standard affine matrices are usually T @ x.
        # grid_sample expects a theta matrix of shape (N, 3, 4).
        # The coordinate system is [-1, 1].
        # Let's adjust. If we want to rotate by R and scale by S, the coordinate mapping from output to input is inv(S) @ inv(R).
        # But for small random perturbations, R @ S vs inv(R) @ inv(S) just changes the direction of random transform,
        # which is symmetric for uniform distributions centered at 0/1. So we can use the matrix directly or its inverse.
        # Let's use the forward matrix construction for simplicity, accepting it might be the inverse transform effectively.
        
        M_3x3 = R @ S
        
        # Translation
        # grid_sample shift is in [-1, 1] range.
        tx = random.uniform(-self.affine_translate, self.affine_translate)
        ty = random.uniform(-self.affine_translate, self.affine_translate)
        tz = random.uniform(-self.affine_translate, self.affine_translate)
        
        # Construct 3x4 matrix
        # theta = [[m00, m01, m02, tx], [m10, m11, m12, ty], [m20, m21, m22, tz]]
        M_3x4 = torch.cat([M_3x3, torch.tensor([[tx], [ty], [tz]], dtype=torch.float32)], dim=1)
        
        # F.affine_grid expects batch (N, 3, 4)
        theta = M_3x4.unsqueeze(0) # (1, 3, 4)
        
        # Prepare volume
        orig_dim = volume.dim()
        if orig_dim == 3:
             # Add batch and channel: (1, 1, D, H, W)
             x = volume.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 4:
             # Add batch: (1, C, D, H, W)
             x = volume.unsqueeze(0)
        else:
             raise ValueError("Volume must be 3D or 4D")

        # Grid sample
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # Squeeze back
        if orig_dim == 3:
            out = out.squeeze(0).squeeze(0)
        elif orig_dim == 4:
            out = out.squeeze(0)
            
        # Check for NaN after affine
        if torch.isnan(out).any():
            raise ValueError("NaN detected after affine transform")
            
        return out

    def normalize(self, volume):
        # Check for NaN before normalization
        if torch.isnan(volume).any():
            # raise VolumeNormalizationError(
            #     f"NaN detected before normalization. Volume shape: {volume.shape}"
            # )
            return torch.zeros_like(volume)
        # Use torch.quantile (percentile)
        # Note: quantile works on float tensors
        if not volume.is_floating_point():
            volume = volume.float()

        q2 = torch.quantile(volume, 0.0001)
        q98 = torch.quantile(volume, 0.999)

        # Handle degenerate cases
        if q98 == 0 or torch.isnan(q98):
            # User prefers to skip these volumes rather than stop training.
            # We approximate "skip" by returning a zero volume so the batch
            # contributes minimally to the loss instead of crashing the DataLoader.
            return torch.zeros_like(volume)

        # Clip (in-place)
        volume.clamp_(min=q2, max=q98)

        # Normalize (in-place)
        # volume = (volume - q2) / (q98 - q2)
        volume.sub_(q2).div_(q98 - q2)
        
        # Check for NaN after normalization
        if torch.isnan(volume).any():
            # raise VolumeNormalizationError(
            #     f"NaN detected after normalization."
            # )
            return torch.zeros_like(volume)
        return volume

    def _process_volume(self, v, target_size):
        # Helper to process a volume (resize, normalize)
        # Step 2: Resize
        v_out = self.resize(v, target_size=target_size)
        # Step 3: Normalize    
        v_out = self.normalize(v_out)
        
        # Step 4: Augmentation (if training)
        # User requested to SKIP affine for now
        # if self.is_train:
        #     v = self.random_affine(v)
            
        # Ensure single channel output
        if v_out.dim() == 4 and v_out.shape[0] == 3:
            v_out = v_out[0:1] # Keep 1st channel, shape (1, D, H, W)

        # Ensure correct shape for 3D ViT: should be (C, D, H, W) where C=1
        if v_out.dim() == 3:
            v_out = v_out.unsqueeze(0) # (D, H, W) -> (1, D, H, W)
            
        if v_out.dim() != 4:
            raise ValueError(f"Expected 4D output (C, D, H, W), got {v_out.shape}")
        if v_out.shape[0] != 1:
            raise ValueError(f"Expected single channel output, got {v_out.shape[0]} channels")
            
        return v_out

    def __call__(self, volume):
        # volume: (C, D, H, W) or (D, H, W)
        # Ensure input is Tensor
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).float()
        elif not isinstance(volume, torch.Tensor):
            # Try to convert whatever it is
            volume = torch.as_tensor(volume).float()
            
        # Check for NaN in input explicitly at the start
        if torch.isnan(volume).any():
             raise ValueError("Input volume contains NaN values")
            
        # Store original for example saving
        original_volume = volume.clone() if self.save_examples and self._examples_saved < self._max_examples else None
        
        # Step 1: Apply fixed crop if coordinates provided (for registered images)
        v_cropped = self.crop_fixed(volume)
        
        # Save crop example if requested
        if self.save_examples and self._examples_saved < self._max_examples and self.crop_coords is not None:
            self.save_crop_example(original_volume, v_cropped, self._examples_saved)
            self._examples_saved += 1
        
        # Process main image
        v_main = self._process_volume(v_cropped.clone(), self.image_size)

        if self.clip_image_size is not None:
            # Process separate CLIP image
            v_clip = self._process_volume(v_cropped.clone(), self.clip_image_size)
            
            # Final NaN check
            if torch.isnan(v_main).any() or torch.isnan(v_clip).any():
                 raise ValueError("NaN detected in output volume(s)")
                 
            return {'image': v_main, 'clip_image': v_clip}

        # Final NaN check
        if torch.isnan(v_main).any():
            raise ValueError("NaN detected in final transform output")
            
        return v_main


def volume_transform_v2(
        cfg: PreprocessCfg,
        is_train: bool,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    # Accepts the same config as image_transform_v2, but returns a Volume3DTransform
    # Only uses relevant fields for 3D
    mean = cfg.mean if hasattr(cfg, 'mean') else 0.0
    std = cfg.std if hasattr(cfg, 'std') else 1.0
    image_size = cfg.size if hasattr(cfg, 'size') else 128
    # Augmentation config
    p_flip = 0.5
    affine_degrees = 15
    affine_translate = 0.1
    affine_scale = (0.9, 1.1)
    crop_coords = None
    with_skull = False
    if aug_cfg is not None:
        if isinstance(aug_cfg, dict):
            p_flip = aug_cfg.get('p_flip', p_flip)
            affine_degrees = aug_cfg.get('affine_degrees', affine_degrees)
            affine_translate = aug_cfg.get('affine_translate', affine_translate)
            affine_scale = aug_cfg.get('affine_scale', affine_scale)
            crop_coords = aug_cfg.get('crop_coords', crop_coords)
            with_skull = aug_cfg.get('with_skull', with_skull)
        elif isinstance(aug_cfg, AugmentationCfg):
            # You can extend this if you add 3D-specific fields to AugmentationCfg
            pass
    return Volume3DTransform(
        image_size=image_size,
        is_train=is_train,
        mean=mean,
        std=std,
        p_flip=p_flip,
        affine_degrees=affine_degrees,
        affine_translate=affine_translate,
        affine_scale=affine_scale,
        crop_coords=crop_coords,
        with_skull=with_skull,
    )
