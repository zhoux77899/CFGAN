"""
This module provides configuration file parsing and configuration class-related functions.
It supports reading configuration information from INI and YAML files,
and converting them into corresponding configuration objects.
"""
import ast
import configparser
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import yaml

PROJ_ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def get_value(value: Any, type_hint: str) -> Any:
    if "os.PathLike" in type_hint:
        return os.path.join(PROJ_ROOT_DIR, value)
    if type_hint.lower().startswith("dict"):
        try:
            return ast.literal_eval(value)
        except ValueError:
            return value
    if type_hint.lower().startswith("bool"):
        return {"true": True, "false": False}[str(value).lower()]
    for convert in (int, float, str):
        try:
            return convert(value)
        except ValueError:
            continue
    return value


def dict_from_ini_file(
        ini_file: Union[str, os.PathLike],
        section_name: str,
        kv_map: dict[str, str]
) -> dict[str, Any]:
    config = configparser.ConfigParser()
    config.read(ini_file)
    section = config[section_name]

    config_dict = {}
    for k, v in kv_map.items():
        if k in section:
            config_dict[k] = get_value(section[k], v)

    return config_dict


def dict_from_yaml_file(
        yaml_file: Union[str, os.PathLike],
        section_name: str,
        kv_map: dict[str, str],
) -> dict[str, Any]:
    with open(yaml_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    section = config[section_name]

    config_dict = {}
    for k, v in kv_map.items():
        if k in section:
            config_dict[k] = get_value(section[k], v)

    return config_dict


class BaseConfig(ABC):
    """Base class for different configurations.
    """

    config_type: str = ""
    kv_map: dict[str, str] = {}

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_ini_file(cls, ini_file: Union[str, os.PathLike]) -> "BaseConfig":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_yaml_file(cls, yaml_file: Union[str, os.PathLike]) -> "BaseConfig":
        raise NotImplementedError


class CFGANDatasetConfig(BaseConfig):
    """Configuration of CFGAN dataset
    """

    config_type = "cfgan_dataset"
    kv_map = {
        "std_dir": "Union[str, os.PathLike]",
        "loc_dir": "Union[str, os.PathLike]",
        "psf_dir": "Union[str, os.PathLike]",
        "img_dir": "Union[str, os.PathLike]",
    }

    def __init__(self, **kwargs: Any) -> None:
        self.std_dir: Union[str, os.PathLike] = os.path.realpath(kwargs.pop("std_dir", ""))
        self.loc_dir: Union[str, os.PathLike] = os.path.realpath(kwargs.pop("loc_dir", ""))
        self.psf_dir: Union[str, os.PathLike] = os.path.realpath(kwargs.pop("psf_dir", ""))
        self.img_dir: Union[str, os.PathLike] = os.path.realpath(kwargs.pop("img_dir", ""))

    @classmethod
    def from_ini_file(cls, ini_file: Union[str, os.PathLike]) -> "CFGANDatasetConfig":
        config_dict = dict_from_ini_file(
            ini_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)

    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str, os.PathLike]) -> "CFGANDatasetConfig":
        config_dict = dict_from_yaml_file(
            yaml_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)


class TerminatorDatasetConfig(BaseConfig):
    """Configuration of Terminator dataset
    """

    config_type = "terminator_dataset"
    kv_map = {
        "residual_dir": "Union[str, os.PathLike]",
    }

    def __init__(self, **kwargs: Any) -> None:
        self.residual_dir: Union[str, os.PathLike] = os.path.realpath(kwargs.pop("residual_dir", ""))

    @classmethod
    def from_ini_file(cls, ini_file: Union[str, os.PathLike]) -> "TerminatorDatasetConfig":

        config_dict = dict_from_ini_file(
            ini_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)

    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str, os.PathLike]) -> "TerminatorDatasetConfig":
        config_dict = dict_from_yaml_file(
            yaml_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)


class FilterConfig(BaseConfig):
    """Configuration of filter
    """

    config_type = "filter"
    kv_map = {
        "filter_args": "Dict[str, Any]",
    }

    def __init__(self, **kwargs: Any) -> None:
        self.filter_args: Dict[str, Any] = kwargs.pop("filter_args", {})

    @classmethod
    def from_ini_file(cls, ini_file: Union[str, os.PathLike]) -> "FilterConfig":
        config_dict = dict_from_ini_file(
            ini_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)

    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str, os.PathLike]) -> "FilterConfig":
        config_dict = dict_from_yaml_file(
            yaml_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)


class ModelConfig(BaseConfig):
    """Configuration of model
    """

    config_type = "model"
    kv_map = {
        "model_path": "Union[str, os.PathLike]",
    }

    def __init__(self, **kwargs: Any) -> None:
        self.model_path: Union[str, os.PathLike] = os.path.realpath(kwargs.pop("model_path", ""))

    @classmethod
    def from_ini_file(cls, ini_file: Union[str, os.PathLike]) -> "ModelConfig":
        config_dict = dict_from_ini_file(
            ini_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)

    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str, os.PathLike]) -> "ModelConfig":
        config_dict = dict_from_yaml_file(
            yaml_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)


class EnvConfig(BaseConfig):
    """Configuration of ENV
    """

    config_type = "env"
    kv_map = {
        "pitch": "float",
        "limit": "float",
        "bias": "float",
        "std_size": "int",
        "std_step": "float",
        "clip_size": "int",
        "clip_bias": "int",
        "tiny": "float",
        "threshold": "float"
    }

    def __init__(self, **kwargs: Any) -> None:
        self.pitch: float = kwargs.pop("pitch", 0.08)
        self.limit: float = kwargs.pop("limit", 4.0)
        self.bias: float = kwargs.pop("bias", 4.0)
        self.std_size: int = kwargs.pop("std_size", 1282)
        self.std_step: float = kwargs.pop("std_step", 0.01)
        self.clip_size: int = kwargs.pop("clip_size", 256)
        self.clip_bias: int = kwargs.pop("clip_bias", 2)
        self.tiny: float = kwargs.pop("tiny", 1e-9)
        self.threshold: float = kwargs.pop("threshold", 0.2)

    @classmethod
    def from_ini_file(cls, ini_file: Union[str, os.PathLike]) -> "EnvConfig":
        config_dict = dict_from_ini_file(
            ini_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)

    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str, os.PathLike]) -> "EnvConfig":
        config_dict = dict_from_yaml_file(
            yaml_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)


class TrainerConfig(BaseConfig):
    """Configuration of trainer
    """

    config_type = "trainer"
    kv_map = {
        "num_epochs": "int",
        "num_batches": "int",
        "num_workers": "int",
        "num_gpus": "int",
        "learning_rate": "float",
        "adam_beta1": "float",
        "adam_beta2": "float",
        "adam_eps": "float",
        "save_dir": "Union[str, os.PathLike]",
        "temp_dir": "Union[str, os.PathLike]"
    }

    def __init__(self, **kwargs: Any) -> None:
        self.num_epochs: int = kwargs.pop("num_epochs", 100)
        self.num_batches: int = kwargs.pop("num_batches", 10)
        self.num_workers: int = kwargs.pop("num_workers", 0)
        self.num_gpus: int = kwargs.pop("num_gpus", 1)
        self.learning_rate: float = kwargs.pop("learning_rate", 0.0002)
        self.adam_beta1: float = kwargs.pop("adam_beta1", 0.9)
        self.adam_beta2: float = kwargs.pop("adam_beta2", 0.999)
        self.adam_eps: float = kwargs.pop("adam_eps", 1e-08)
        self.save_dir: Union[str, os.PathLike] = os.path.realpath(kwargs.pop("save_dir", ""))
        self.temp_dir: Union[str, os.PathLike] = os.path.realpath(kwargs.pop("temp_dir", ""))

    @classmethod
    def from_ini_file(cls, ini_file: Union[str, os.PathLike]) -> "TrainerConfig":
        config_dict = dict_from_ini_file(
            ini_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)

    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str, os.PathLike]) -> "TrainerConfig":
        config_dict = dict_from_yaml_file(
            yaml_file,
            cls.config_type,
            cls.kv_map
        )
        return cls(**config_dict)
