"""
MR-SYNTH: Configuration and Distributed Environment Settings
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from monai.bundle import ConfigParser


def setup_logging(output_dir: str, log_filename: str = "mr_synth.log") -> logging.Logger:
    """Initialize stream and file logger."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("MR-SYNTH")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File handler
    fh = logging.FileHandler(os.path.join(output_dir, log_filename))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def initialize_distributed(args: argparse.Namespace | None = None) -> tuple[int, int, int]:
    """Initialize torch distributed environment if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(seconds=7200),
            )
        return rank, world_size, local_rank
    return 0, 1, 0


def load_config(
    environment_file: str,
    config_file: str,
    network_file: str | None = None,
    extra_configs: list[str] | None = None,
) -> argparse.Namespace:
    """Parse MONAI JSON configuration files into a flat Namespace."""
    parser = ConfigParser()

    all_files = [environment_file, config_file]
    if network_file:
        all_files.append(network_file)
    if extra_configs:
        all_files.extend(extra_configs)

    for cf in all_files:
        if cf and os.path.exists(cf):
            parser.read_config(cf)

    config_dict = parser.get()
    return argparse.Namespace(**config_dict)
