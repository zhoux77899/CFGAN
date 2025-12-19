__version__ = "0.1.0"

import os
import sys

from .models.modeling_cfgan import UNetGenerator, CNNDiscriminator
from .config import CFGANDatasetConfig


root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, ".."))


__all__ = [
    "UNetGenerator",
    "CNNDiscriminator",
    "CFGANDatasetConfig"
]
