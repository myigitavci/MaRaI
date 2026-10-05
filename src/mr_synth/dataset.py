"""
MR-SYNTH: Dataset and Embedding Store Utilities
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any

import monai
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged


class EmbeddingStore:
    """
    Lightweight wrapper around pre-computed MR-CLIP text embeddings PKL.
    Provides O(1) embedding retrieval by sample index.
    """

    def __init__(self, pkl_path: str) -> None:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.embeddings: np.ndarray = data["embeddings"].astype(np.float32)
        self.filepaths: list[str] = data.get("filepaths", [])
        self.texts: list[str] = data.get("texts", [])

    def __len__(self) -> int:
        return len(self.embeddings)

    def get(self, idx: int) -> torch.Tensor:
        """Return a (512,) float32 tensor."""
        return torch.from_numpy(self.embeddings[idx])


def load_filenames_and_indices(data_json_path: str) -> list[dict[str, Any]]:
    """
    Load dataset JSON list where each entry has {"image": "..._emb.nii.gz", "idx": int}.
    """
    with open(data_json_path, "r") as f:
        js = json.load(f)
    if isinstance(js, dict):
        return js.get("training", js.get("data", []))
    return js


def prepare_data_mrclip(
    train_files: list[dict[str, Any]],
    emb_store: EmbeddingStore,
    embedding_base_dir: str = "./embeddings",
    cache_rate: float = 0.0,
    num_workers: int = 2,
    batch_size: int = 1,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a DataLoader yielding latent batches with MR-CLIP text embeddings.
    """
    def _load_spacing(info_path: str | None) -> torch.Tensor:
        if info_path is not None and os.path.exists(info_path):
            with open(info_path) as f:
                d = json.load(f)
            return torch.FloatTensor(d.get("spacing", [1.0, 1.0, 1.0])) * 1e2
        return torch.FloatTensor([0.94, 0.94, 1.36]) * 1e2

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])

    if cache_rate > 0.0:
        ds = CacheDataset(data=train_files, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
    else:
        ds = Dataset(data=train_files, transform=transforms)

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        embeddings = torch.stack([emb_store.get(item["idx"]) for item in batch])
        spacings = torch.stack([_load_spacing(item.get("info")) for item in batch])
        return {
            "image": images,
            "clip_embedding": embeddings,
            "spacing": spacings,
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
