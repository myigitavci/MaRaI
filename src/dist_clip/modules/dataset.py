import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import namedtuple
import nibabel as nib
from torchvision.transforms import Compose, Pad, CenterCrop, ToTensor, Resize, Normalize, InterpolationMode
from PIL import Image
import scipy.ndimage
import random
# Import Volume3DTransform from clip_transform instead of duplicating
from dist_clip.clip_transform import Volume3DTransform

def classic_normalize(img):
    # img is a torch tensor
    if isinstance(img, torch.Tensor):
        min_val = img.min()
        max_val = img.max()
        if max_val > min_val:
            return (img - min_val) / (max_val - min_val)
        else:
            return img - min_val  # fallback if all values are the same
    else:
        return img
    
# CLIP transforms
transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop((224, 224)),
            #lambda x: x.convert('RGB'),
            ToTensor(),
            classic_normalize,
            #Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])

# Transform without additional normalization (for 3D-normalized data)
transform_no_norm = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop((224, 224)),
            ToTensor(),
            # No classic_normalize - already normalized at 3D level
        ])
def normalize98(volume):
    if np.isnan(volume).any():
        return np.zeros_like(volume)
    q = np.percentile(volume, 98)
    if q == 0:
        return np.zeros_like(volume)
    volume = volume / q
    volume[volume > 1] = 1
    volume[volume < 0] = 0
    return volume

def percentile_clip(input_tensor, reference_tensor=None, p_min=0.01, p_max=99.9, strictlyPositive=True):
    if(reference_tensor == None):
        reference_tensor = input_tensor
    v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile
    if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
        v_min = 0
    output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
    output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]
    return output_tensor

def load_nifti_slices(filepath, slice_start=None, slice_end=None, use_percentile_clip=False, percentile_min=0.01, percentile_max=99.9):
    if os.path.exists(filepath):
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.float32)
        
        # Apply 3D volume normalization first
        if use_percentile_clip:
            # Use percentile clipping for 3D volume normalization
            normalized_data = percentile_clip(data, p_min=percentile_min, p_max=percentile_max) * 255
        else:
            # Use min/max normalization for entire 3D volume
            normalized_data = (data - data.min()) / (data.max() - data.min()) *255
            
        normalized_data = normalized_data.astype(np.uint8)
        if normalized_data.ndim == 3:
            # Use full slice range when slice_start/end are not provided
            if slice_start is None or slice_end is None:
                slc = normalized_data
            else:
                slc = normalized_data[:, :, slice_start:slice_end:2]
            normalized_data = slc
        else:
            raise ValueError(f"NIfTI file {filepath} does not have 3 dimensions.")
        slices = []

        for i in range(normalized_data.shape[2]):
            slice_img = normalized_data[:, :, i]
            # Convert to PIL Image for transforms
            slice_img = Image.fromarray(slice_img)
            # Apply transforms without additional normalization (already normalized at 3D level)
            slice_img = transform_no_norm(slice_img)
            slices.append(slice_img)
        return torch.stack(slices, dim=0)
    else:
        print(f"File {filepath} does not exist.")
        return torch.ones(((slice_end-slice_start)//2, 3, 224, 224))
    

PairItem = namedtuple('PairItem', ['img', 'filepath', 'text', 'label', 'orientation'])

class PairedImageDataset(Dataset):
    def __init__(self, csv_path, slice_start=None, slice_end=None, preload=False, use_percentile_clip=False, percentile_min=0.01, percentile_max=99.9):
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.preload = preload
        self.use_percentile_clip = use_percentile_clip
        self.percentile_min = percentile_min
        self.percentile_max = percentile_max

        df = pd.read_csv(csv_path)
        # Group by pair_id
        self.pair_groups = {}
        for _, row in df.iterrows():
            pair_id = row['pair']
            entry = {
                'filepath': row['filepath'],
                'text': row['text'],
                'label': row['label'],
                'orientation': row['orientation'] if 'orientation' in df.columns else 'axial',
            }
            if pair_id not in self.pair_groups:
                self.pair_groups[pair_id] = []
            self.pair_groups[pair_id].append(entry)

        # Build all possible (src, tgt, pair_id) pairs (src ≠ tgt)
        self.all_pairs = []
        for pair_id, items in self.pair_groups.items():
            n = len(items)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.all_pairs.append((pair_id, i, j))

        # Optional: preload all images into memory (if fits)
        self.image_cache = {}
        if self.preload:
            for pair_id, items in self.pair_groups.items():
                for idx, entry in enumerate(items):
                    key = (pair_id, idx)
                    self.image_cache[key] = load_nifti_slices(entry['filepath'], self.slice_start, self.slice_end, 
                                                           self.use_percentile_clip, self.percentile_min, self.percentile_max)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        pair_id, src_idx, tgt_idx = self.all_pairs[idx]
        src_entry = self.pair_groups[pair_id][src_idx]
        tgt_entry = self.pair_groups[pair_id][tgt_idx]

        if self.preload:
            src_img = self.image_cache[(pair_id, src_idx)]
            tgt_img = self.image_cache[(pair_id, tgt_idx)]
        else:
            src_img = load_nifti_slices(src_entry['filepath'], self.slice_start, self.slice_end, 
                                     self.use_percentile_clip, self.percentile_min, self.percentile_max)
            tgt_img = load_nifti_slices(tgt_entry['filepath'], self.slice_start, self.slice_end,
                                       self.use_percentile_clip, self.percentile_min, self.percentile_max)

        src_item = PairItem(img=src_img, filepath=src_entry['filepath'], text=src_entry['text'], label=src_entry['label'], orientation=src_entry.get('orientation', 'axial'))
        tgt_item = PairItem(img=tgt_img, filepath=tgt_entry['filepath'], text=tgt_entry['text'], label=tgt_entry['label'], orientation=tgt_entry.get('orientation', 'axial'))
        return {
            'pair_id': pair_id,
            'source': src_item,
            'target': tgt_item
        }

class Paired3DImageDataset(Dataset):
    def __init__(self, csv_path, preload=False, transform=None, image_size=128, is_train=True, crop_coords=None, save_examples=False, clip_image_size=None, with_skull=False):
        """
        Args:
            csv_path: Path to CSV file with image pairs
            preload: Whether to preload images into memory
            transform: Volume3DTransform instance (if None, will create one)
            image_size: Size for resizing (default: 128)
            is_train: Whether in training mode (affects augmentation)
            crop_coords: Dict with crop coordinates (d_min, d_max, h_min, h_max, w_min, w_max)
                         For registered images, use same crop coords for all volumes
            save_examples: If True, saves first few cropped examples for verification
            clip_image_size: Optional separate size for CLIP images
            with_skull: If True, use crop coordinates that include skull; if False, use no-skull coordinates
        """
        self.preload = preload
        self.image_size = image_size
        self.is_train = is_train
        self.crop_coords = crop_coords
        self.save_examples = save_examples
        self.clip_image_size = clip_image_size
        self.with_skull = with_skull
        
        # Create transform if not provided
        if transform is None:
            self.transform = Volume3DTransform(
                image_size=image_size, 
                is_train=is_train,
                crop_coords=crop_coords,
                save_examples=save_examples,
                clip_image_size=clip_image_size,
                with_skull=with_skull
            )
        else:
            self.transform = transform

        df = pd.read_csv(csv_path)
        # Group by pair_id
        self.pair_groups = {}
        for _, row in df.iterrows():
            pair_id = row['pair']
            entry = {
                'filepath': row['filepath'],
                'text': row['text'],
                'label': row['label'],
                'orientation': row['orientation'] if 'orientation' in df.columns else 'axial',
            }
            if pair_id not in self.pair_groups:
                self.pair_groups[pair_id] = []
            self.pair_groups[pair_id].append(entry)

        # Build all possible (src, tgt, pair_id) pairs (src ≠ tgt)
        self.all_pairs = []
        for pair_id, items in self.pair_groups.items():
            n = len(items)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.all_pairs.append((pair_id, i, j))
        
        # Optional: preload all images into memory (if fits)
        self.image_cache = {}
        if self.preload:
            for pair_id, items in self.pair_groups.items():
                for idx, entry in enumerate(items):
                    key = (pair_id, idx)
                    self.image_cache[key] = self._load_and_transform_volume(entry['filepath'])

    def _load_and_transform_volume(self, filepath):
        """Load and transform a single volume"""
        img = nib.load(filepath)
        volume = img.get_fdata().astype(np.float32)
        
        # Add channel dimension if needed: (D, H, W) -> (1, D, H, W)
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]  # Add channel dimension
        
        # Apply transform
        volume_transformed = self.transform(volume)
        
        return volume_transformed

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        pair_id, src_idx, tgt_idx = self.all_pairs[idx]
        src_entry = self.pair_groups[pair_id][src_idx]
        tgt_entry = self.pair_groups[pair_id][tgt_idx]

        # Load and transform images
        if self.preload:
            src_res = self.image_cache[(pair_id, src_idx)]
            tgt_res = self.image_cache[(pair_id, tgt_idx)]
        else:
            src_res = self._load_and_transform_volume(src_entry['filepath'])
            tgt_res = self._load_and_transform_volume(tgt_entry['filepath'])

        # Handle potential dictionary return from transform
        if isinstance(src_res, dict):
            src_img = src_res['image']
            src_clip = src_res['clip_image']
        else:
            src_img = src_res
            src_clip = None
            
        if isinstance(tgt_res, dict):
            tgt_img = tgt_res['image']
            tgt_clip = tgt_res['clip_image']
        else:
            tgt_img = tgt_res
            tgt_clip = None

        src_item = PairItem(img=src_img, filepath=src_entry['filepath'], text=src_entry['text'], label=src_entry['label'], orientation=src_entry.get('orientation', 'axial'))
        tgt_item = PairItem(img=tgt_img, filepath=tgt_entry['filepath'], text=tgt_entry['text'], label=tgt_entry['label'], orientation=tgt_entry.get('orientation', 'axial'))
        
        ret = {
            'pair_id': pair_id,
            'source': src_item,
            'target': tgt_item
        }
        
        if src_clip is not None:
            ret['source_clip'] = src_clip
        if tgt_clip is not None:
            ret['target_clip'] = tgt_clip
            
        return ret

# Volume3DTransform is now imported from clip_transform.py above
# This provides brain cropping with MONAI and consistent preprocessing

class PairedSliceDataset(Dataset):
    def __init__(self, csv_path, slice_start=140, slice_end=142, preload=False):
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.preload = preload

        df = pd.read_csv(csv_path,on_bad_lines='warn')
        # Group by pair_id
        self.pair_groups = {}
        for _, row in df.iterrows():
            pair_id = row['pair']
            entry = {
                'filepath': row['filepath'],
                'text': row['text'],
                'label': row['label'],
                'orientation': row['orientation'] if 'orientation' in df.columns else 'axial',
            }
            if pair_id not in self.pair_groups:
                self.pair_groups[pair_id] = []
            self.pair_groups[pair_id].append(entry)

        # Preload images if requested
        self.image_cache = {}
        for pair_id, items in self.pair_groups.items():
            for idx, entry in enumerate(items):
                key = (pair_id, idx)
                if self.preload:
                    self.image_cache[key] = load_nifti_slices(entry['filepath'], self.slice_start, self.slice_end)
                else:
                    self.image_cache[key] = None  # Placeholder

        # Build all possible (src_img, src_slice_idx, tgt_img, tgt_slice_idx, pair_id) pairs (src_img ≠ tgt_img)
        self.all_slice_pairs = []
        for pair_id, items in self.pair_groups.items():
            n = len(items)
            # Load one image to get number of slices
            num_slices = None
            for idx, entry in enumerate(items):
                if self.preload and self.image_cache[(pair_id, idx)] is not None:
                    num_slices = self.image_cache[(pair_id, idx)].shape[0]
                    break
                else:
                    img = nib.load(entry['filepath'])
                    data = img.get_fdata()
                    num_slices = data[:, :, self.slice_start:self.slice_end:2].shape[2]
                    break
            for i in range(n):
                for j in range(n):
                    if i != j:
                        for s in range(num_slices):
                            for t in range(num_slices):
                                self.all_slice_pairs.append((pair_id, i, s, j, t))

    def __len__(self):
        return len(self.all_slice_pairs)

    def __getitem__(self, idx):
        pair_id, src_img_idx, src_slice_idx, tgt_img_idx, tgt_slice_idx = self.all_slice_pairs[idx]
        src_entry = self.pair_groups[pair_id][src_img_idx]
        tgt_entry = self.pair_groups[pair_id][tgt_img_idx]

        # Load or get cached slices
        if self.preload and self.image_cache[(pair_id, src_img_idx)] is not None:
            src_slices = self.image_cache[(pair_id, src_img_idx)]
        else:
            src_slices = load_nifti_slices(src_entry['filepath'], self.slice_start, self.slice_end)
        if self.preload and self.image_cache[(pair_id, tgt_img_idx)] is not None:
            tgt_slices = self.image_cache[(pair_id, tgt_img_idx)]
        else:
            tgt_slices = load_nifti_slices(tgt_entry['filepath'], self.slice_start, self.slice_end)

        src_slice = src_slices[src_slice_idx]  # shape: (1, 224, 224)
        tgt_slice = tgt_slices[tgt_slice_idx]  # shape: (1, 224, 224)

        src_item = PairItem(img=src_slice, filepath=src_entry['filepath'], text=src_entry['text'], label=src_entry['label'], orientation=src_entry.get('orientation', 'axial'))
        tgt_item = PairItem(img=tgt_slice, filepath=tgt_entry['filepath'], text=tgt_entry['text'], label=tgt_entry['label'], orientation=tgt_entry.get('orientation', 'axial'))
        return {
            'pair_id': pair_id,
            'source': src_item,
            'target': tgt_item
        }

class PairedPNGSliceDataset(Dataset):
    def __init__(self, csv_path, preload=False, transform=None):
        self.preload = preload
        self.transform = transform
        df = pd.read_csv(csv_path)
        # Group by pair_id and by slice number
        self.pair_groups = {}
        for _, row in df.iterrows():
            pair_id = row['pair']
            filepath = row['filepath']
            # Extract slice number from filename (assumes ..._slc123.png)
            slc = int(os.path.splitext(filepath)[0].split('_slc')[-1])
            entry = {
                'filepath': filepath,
                'text': row['text'],
                'label': row['label'],
                'slice': slc
            }
            if pair_id not in self.pair_groups:
                self.pair_groups[pair_id] = {}
            if slc not in self.pair_groups[pair_id]:
                self.pair_groups[pair_id][slc] = []
            self.pair_groups[pair_id][slc].append(entry)

        # Build all possible (src, tgt, pair_id, slice) pairs (src ≠ tgt, same slice)
        self.all_pairs = []
        for pair_id, slices in self.pair_groups.items():
            for slc, items in slices.items():
                n = len(items)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            self.all_pairs.append((pair_id, slc, i, j))

        # Optional: preload all images into memory (if fits)
        self.image_cache = {}
        if self.preload:
            for pair_id, slices in self.pair_groups.items():
                for slc, items in slices.items():
                    for idx, entry in enumerate(items):
                        key = (pair_id, slc, idx)
                        self.image_cache[key] = self.load_png(entry['filepath'])

    def load_png(self, filepath):
        img = Image.open(filepath).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        return img.unsqueeze(0) if img.ndim == 2 else img  # (1, H, W)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        pair_id, slc, src_idx, tgt_idx = self.all_pairs[idx]
        src_entry = self.pair_groups[pair_id][slc][src_idx]
        tgt_entry = self.pair_groups[pair_id][slc][tgt_idx]

        if self.preload and (pair_id, slc, src_idx) in self.image_cache:
            src_img = self.image_cache[(pair_id, slc, src_idx)]
        else:
            src_img = self.load_png(src_entry['filepath'])
        if self.preload and (pair_id, slc, tgt_idx) in self.image_cache:
            tgt_img = self.image_cache[(pair_id, slc, tgt_idx)]
        else:
            tgt_img = self.load_png(tgt_entry['filepath'])

        src_item = PairItem(img=src_img, filepath=src_entry['filepath'], text=src_entry['text'], label=src_entry['label'], orientation=src_entry.get('orientation', 'axial'))
        tgt_item = PairItem(img=tgt_img, filepath=tgt_entry['filepath'], text=tgt_entry['text'], label=tgt_entry['label'], orientation=tgt_entry.get('orientation', 'axial'))
        return {
            'pair_id': pair_id,
            'slice': slc,
            'source': src_item,
            'target': tgt_item
        }
