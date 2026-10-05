"""
MR-SYNTH: Core Model and MONAI Utilities
"""

from __future__ import annotations

import copy
import json
import math
import os
from argparse import Namespace
from typing import Any

import numpy as np
import torch
from monai.bundle import ConfigParser
from monai.utils import optional_import

nib, _ = optional_import("nibabel")


def define_instance(args: Namespace | dict, key: str) -> Any:
    """
    Instantiate an object or network from a configuration dictionary or namespace.
    Supports MONAI ConfigParser instantiation syntax (_target_).
    """
    if isinstance(args, Namespace):
        config_dict = vars(args)
    else:
        config_dict = args

    parser = ConfigParser(config_dict)
    return parser.get_parsed_content(key, instantiate=True)


def dynamic_infer(
    inferer: Any,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    autoencoder: torch.nn.Module | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Perform sliding-window or direct inference dynamically handling VAE encode/decode.
    """
    if autoencoder is not None:
        return inferer(inputs=inputs, network=model, autoencoder=autoencoder, **kwargs)
    return inferer(inputs=inputs, network=model, **kwargs)
