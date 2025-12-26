import torch
import random
import numpy as np
import os
from transformers import set_seed
import torch.distributed as dist

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    # For multi-GPU setups
    # CuDNN settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def line_count(filename: str) -> int:
    """Counts the number of lines in a file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.barrier()
    dist.destroy_process_group()