import ast
import json
import logging
import math
import os
import random
import sys
import re

import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None,distance=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.labels = df['label'].tolist() if 'label' in df.columns else None
        self.transforms = transforms
        self.tokenize = tokenizer
        self.distance = distance
        # Extract 3D image ID (removing slice index)
        self.image_groups = defaultdict(list)
        for idx, filepath in enumerate(self.images):
            image_id = "_".join(filepath.split("_")[:-1])  # Remove "_sliceXXX.png"
            self.image_groups[image_id].append(idx)
        
        logging.info(f"Loaded {len(self.images)} slices across {len(self.image_groups)} unique 3D images.")
    def extract_times(self, caption):
        # Find all parenthesis contents
        matches = re.findall(r'\(([^()]*)\)', caption)
        if matches:
            # Extract the numerical values from the last match
            values = re.findall(r'\d+\.\d+|\d+', matches[-1])
            if len(values) >= 2:
                echo_time = float(values[0])
                repetition_time = float(values[1])
                return echo_time, repetition_time
        return None, None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #logging.info(f"{str(idx)} , {str(self.images[idx])}")
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        labels = self.labels[idx] if self.labels is not None else None
        if self.distance:
            echo_time, repetition_time = self.extract_times(self.captions[idx])
            #print(self.captions[idx])
            #print(echo_time, repetition_time)
            return images, texts, labels, echo_time, repetition_time
        return images, texts, labels



class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        distance=args.distance
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "csv-unique-sampler":
        return get_csv_dataset_unique_sampler
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "tabular":
        return get_tabular_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
class UniqueLabelSampler(torch.utils.data.Sampler):
    """
    A PyTorch sampler that ensures no batch contains samples with the same label.
    
    - Ensures each batch contains samples with unique labels.
    - Samples within each label group are shuffled.
    - The order of label groups is also shuffled per epoch.
    - All samples are used in an epoch without repetition.
    - Supports both distributed and non-distributed training.
    """
    def __init__(self, dataset, batch_size, num_replicas=1, rank=0, shuffle=True):
        """
        Initializes the sampler.
        
        Args:
            dataset (Dataset): The dataset containing samples with labels.
            batch_size (int): The number of samples per batch.
            num_replicas (int): Number of distributed replicas (for multi-GPU training).
            rank (int): Rank of the current process (for distributed training).
            shuffle (bool): Whether to shuffle labels and samples.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0  # Stores current epoch
        self.is_distributed = num_replicas > 1  # Check if distributed training is enabled
        
        # Group samples by their label
        self.label_groups = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.label_groups[label].append(idx)
        
        # Store unique labels
        self.labels = list(self.label_groups.keys())
    
    def set_epoch(self, epoch):
        """Sets the current epoch to ensure different shuffling each epoch."""
        self.epoch = epoch
    
    def __iter__(self):
        """
        Generates an iterator for batch sampling, ensuring no batch has samples with the same label.
        """
        # Set random seed for reproducibility per epoch
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.labels)  # Shuffle the labels
        
        # Shuffle samples within each label group
        for label in self.labels:
            random.shuffle(self.label_groups[label])
        
        # Flatten samples while ensuring each batch contains unique labels
        grouped_samples = [self.label_groups[label] for label in self.labels]
        max_samples = max(len(s) for s in grouped_samples)  # Get max samples per label
        batch_samples = []
        
        for i in range(max_samples):
            batch = []
            for label_samples in grouped_samples:
                if i < len(label_samples):
                    batch.append(label_samples[i])
                if len(batch) == self.batch_size:
                    batch_samples.extend(batch)
                    batch = []
            if batch:  # Append remaining samples if not full batch
                batch_samples.extend(batch)
        
        # Handle distributed training if enabled
        if self.is_distributed:
            batch_samples = batch_samples[self.rank::self.num_replicas]
        
        return iter(batch_samples)
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)
class Unique3DSampler(torch.utils.data.Sampler):
    """
    A PyTorch sampler that ensures no batch contains slices from the same 3D image.
    
    - Ensures each batch contains slices from different 3D images.
    - Slices within each 3D image are shuffled.
    - The order of 3D images is also shuffled per epoch.
    - All slices are used in an epoch without repetition.
    - Supports both distributed and non-distributed training.
    """
    def __init__(self, dataset, batch_size, num_replicas=1, rank=0, shuffle=True):
        """
        Initializes the sampler.
        
        Args:
            dataset (Dataset): The dataset containing slices of 3D images.
            batch_size (int): The number of slices per batch.
            num_replicas (int): Number of distributed replicas (for multi-GPU training).
            rank (int): Rank of the current process (for distributed training).
            shuffle (bool): Whether to shuffle images and slices.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0  # Stores current epoch
        self.is_distributed = num_replicas > 1  # Check if distributed training is enabled
        
        # Group slices by their 3D image ID
        self.image_slices = defaultdict(list)
        for idx, path in enumerate(dataset.images):
            image_id = "_".join(path.split("_")[:-1])  # Extract 3D image ID from filename
            self.image_slices[image_id].append(idx)
        
        # Store unique 3D image IDs
        self.image_ids = list(self.image_slices.keys())
    
    def set_epoch(self, epoch):
        """Sets the current epoch to ensure different shuffling each epoch."""
        self.epoch = epoch
    
    def __iter__(self):
        """
        Generates an iterator for batch sampling, ensuring no batch has slices from the same 3D image.
        """
        # Set random seed for reproducibility per epoch
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.image_ids)  # Shuffle the 3D images
        
        # Shuffle slices within each 3D image
        for img_id in self.image_ids:
            random.shuffle(self.image_slices[img_id])
        
        # Flatten slices while ensuring each batch contains unique 3D images
        grouped_slices = [self.image_slices[img_id] for img_id in self.image_ids]
        max_slices = max(len(s) for s in grouped_slices)  # Get max slices per 3D image
        batch_slices = []
        
        for i in range(max_slices):
            batch = []
            for img_slices in grouped_slices:
                if i < len(img_slices):
                    batch.append(img_slices[i])
                if len(batch) == self.batch_size:
                    batch_slices.extend(batch)
                    batch = []
            if batch:  # Append remaining slices if not full batch
                batch_slices.extend(batch)
        
        # Handle distributed training if enabled
        if self.is_distributed:
            batch_slices = batch_slices[self.rank::self.num_replicas]
        
        return iter(batch_slices)
    
    def __len__(self):
        """Returns the total number of slices in the dataset."""
        return len(self.dataset)


def get_csv_dataset_unique_sampler(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename, "Input CSV file is required."

    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )

    sampler = UniqueLabelSampler(dataset, args.batch_size, num_replicas=args.world_size, rank=args.rank)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    num_samples = len(dataset)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class CsvTabularDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, sep="\t", one_hot_encode=False, corruption_rate=0.0):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        
        # Identify feature columns (all except filepath and label)
        self.images = df[img_key].tolist()
        self.labels = df['label'].tolist() if 'label' in df.columns else None
        self.feature_keys = [col for col in df.columns if col not in [img_key, 'label']]
        
        # Identify numerical and categorical columns
        numerical_columns = {"Echo Time", "Repetition Time", "Flip Angle", "Inversion Time"}
        categorical_columns = set(self.feature_keys) - numerical_columns
        
        # Convert numerical columns to float
        df[numerical_columns] = df[numerical_columns].astype(float)
        
        # Compute field lengths for one-hot encoding
        self.field_lengths = {
            col: df[col].nunique() if col in categorical_columns else 1 for col in self.feature_keys
        }
        
        self.features = df[self.feature_keys].values  # Extract features
        self.transforms = transforms
        self.one_hot_encode = one_hot_encode
        self.corruption_rate = corruption_rate
        
        # Generate empirical marginal distributions for corruption
        self.marginal_distributions = {
            col: df[col].tolist() for col in self.feature_keys
        }
        
        # Extract 3D image ID (removing slice index)
        self.image_groups = defaultdict(list)
        for idx, filepath in enumerate(self.images):
            image_id = "_".join(filepath.split("_")[:-1])  # Remove "_sliceXXX.png"
            self.image_groups[image_id].append(idx)
        
        logging.info(f"Loaded {len(self.images)} slices across {len(self.image_groups)} unique 3D images.")
    
    def __len__(self):
        return len(self.images)
    
    def corrupt_features(self, features):
        """Randomly corrupts a fraction of the features by replacing them with sampled values from their marginal distributions."""
        features = features.clone()
        indices = random.sample(range(len(features)), int(len(features) * self.corruption_rate))
        for i in indices:
            feature_key = self.feature_keys[i]
            features[i] = random.choice(self.marginal_distributions[feature_key])
        return features
    
    def one_hot_encode_features(self, features):
        """One-hot encodes the tabular features if required."""
        if not self.one_hot_encode:
            return features
        
        encoded = []
        for i, col in enumerate(self.feature_keys):
            length = self.field_lengths[col]
            if length == 1:
                encoded.append(torch.tensor(features[i], dtype=torch.float).unsqueeze(0))
            else:
                encoded.append(torch.nn.functional.one_hot(torch.tensor(features[i], dtype=torch.long), num_classes=length))
        return torch.cat(encoded)
    
    def __getitem__(self, idx):
        image = self.transforms(Image.open(str(self.images[idx])))
        tabular_data = torch.tensor(self.features[idx], dtype=torch.float)
        tabular_data = self.corrupt_features(tabular_data)
        tabular_data = self.one_hot_encode_features(tabular_data)
        label = self.labels[idx] if self.labels is not None else None
        return image, tabular_data, label

def get_tabular_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename, "Input CSV file is required."

    dataset = CsvTabularDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        sep=args.csv_separator,
        one_hot_encode=args.one_hot_tabular,
        corruption_rate=args.corruption_rate
    )
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)