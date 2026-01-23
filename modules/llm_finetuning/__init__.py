"""
LLM Fine-tuning Module - Package init file
"""
from . import (
    lora_trainer as lora,
    peft_manager as peft
)

# Import functions/classes for easier access
try:
    from .lora_trainer import LoRATrainer, merge_lora_adapter
except ImportError:
    pass

try:
    from .peft_manager import PEFTManager, merge_multiple_adapters, create_lora_config
except ImportError:
    pass

__all__ = [
    # LoRA trainer
    "LoRATrainer",
    "merge_lora_adapter",
    # PEFT manager
    "PEFTManager", 
    "merge_multiple_adapters",
    "create_lora_config"
]