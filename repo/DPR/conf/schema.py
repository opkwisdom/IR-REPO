# This module defines configuration schemas for various components of the DPR system.
# It helps modern IDE to provide better autocompletion and type checking.

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

### =========== Base Configurations ===========

@dataclass
class DataConfig:
    """Configuration for data processing."""
    input_path: str
    output_path: str
    batch_size: int = 32
    shuffle: bool = True

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    model_name: str
    hidden_layers: int = 2
    dropout_rate: float = 0.1
    learning_rate: float = 0.001


@dataclass
class BaseConfig:
    """Base configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42


### =========== Specific Configurations ===========
@dataclass
class NegativeMiningConfig(BaseConfig):
    """Configuration for negative mining."""
    num_hard_negatives: int = 5
    num_random_negatives: int = 10
    use_dynamic_negatives: bool = False


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training."""
    epochs: int = 10
    validation_split: float = 0.2
    early_stopping: bool = True
    early_stopping_patience: int = 3


@dataclass
class IndexingConfig(BaseConfig):
    """Configuration for indexing."""
    index_type: str = "flat"
    use_gpu: bool = False
    batch_size: int = 512


@dataclass
class RetrievalConfig(BaseConfig):
    """Configuration for retrieval."""
    top_k: int = 10
    similarity_metric: str = "cosine"
    rerank: bool = False


# Register configurations with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="negative_mining_config", node=NegativeMiningConfig)
cs.store(name="training_config", node=TrainingConfig)
cs.store(name="indexing_config", node=IndexingConfig)
cs.store(name="retrieval_config", node=RetrievalConfig)