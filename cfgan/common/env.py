import os
from dataclasses import dataclass
from typing import Tuple

from cfgan.config import EnvConfig

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config.ini")
env_config = EnvConfig.from_ini_file(CONFIG_FILE)


@dataclass
class EnvVar:
    """
    Environment variables
    """

    # camera
    pitch: float = env_config.pitch

    # molecules
    limit: float = env_config.limit
    bias: float = env_config.bias

    # images
    std_size: Tuple[int, int] = (env_config.std_size, env_config.std_size)
    std_step: float = env_config.std_step
    clip_size: Tuple[int, int] = (env_config.clip_size, env_config.clip_size)
    clip_bias: Tuple[int, int] = (env_config.clip_bias, -env_config.clip_bias)

    # utils
    tiny: float = env_config.tiny
    threshold: float = env_config.threshold


ENV = EnvVar()
