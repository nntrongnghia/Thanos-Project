from typing import Dict, List
import torch
from torch.utils.data import DataLoader
import numpy as np
from abc import ABC, abstractmethod
import importlib
import os

class BaseTrainConfig(ABC):
    PROJECT_NAME = ""
    EXPE_NAME = ""
    @abstractmethod
    def model_config(self) -> Dict:
        """Return a dict which will be passed to model constructor"""
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return a dict which will be passed to dataset constructor"""
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return a dict which will be passed to dataset constructor"""
        pass

    @abstractmethod
    def trainer_config(self) -> Dict:
        """Return a dict which will be passed to Lightning Module constructor"""
        pass

        
def load_config(config_path) -> BaseTrainConfig:
    """
    Parameters
    ----------
    config_path: str
        Relative path to config py file
    
    Returns
    -------
    a subclass of BaseTrainConfig
    """
    thanos_root = os.path.join(os.path.dirname(__file__), "..", "..")
    abs_config_path = os.path.join(thanos_root, config_path)
    spec = importlib.util.spec_from_file_location("config", abs_config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config