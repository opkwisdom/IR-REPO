import hydra
from omegaconf import DictConfig, OmegaConf

from conf import schema
from conf.schema import NegativeMiningConfig


@hydra.main(version_base=None, config_path="conf", config_name="negative_mining")
def main(cfg: NegativeMiningConfig):
    print("Negative Mining Configuration:")
    print(cfg)