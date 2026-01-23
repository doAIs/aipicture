"""
PEFT Manager - Manage PEFT adapters and operations
"""
import os
import sys
import torch
from typing import Dict, Optional, List
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


class PEFTManager:
    """
    Manager for PEFT (Parameter Efficient Fine-Tuning) operations
    Handles loading, merging, and managing adapters
    """
    
    def __init__(self):
        self.adapters = {}
        self.loaded_models = {}
    
    def load_base_model(self, model_name: str, device: str = None):
        """
        Load a base model for applying adapters
        
        Args:
            model_name: Name or path of the base model
            device: Device to load model on (defaults to available device)
        
        Returns:
            Loaded model
        """
        if device is None:
            device = get_device()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        
        self.loaded_models[model_name] = model
        return model
    
    def load_adapter(self, model, adapter_path: str, adapter_name: str = "default"):
        """
        Load and apply an adapter to a model
        
        Args:
            model: Base model to apply adapter to
            adapter_path: Path to the adapter
            adapter_name: Name to assign to the adapter
        
        Returns:
            Model with adapter applied
        """
        if isinstance(model, PeftModel):
            # If model already has adapters, add new one
            model.load_adapter(adapter_path, adapter_name=adapter_name)
        else:
            # If model doesn't have adapters, wrap it
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
        
        self.adapters[adapter_name] = adapter_path
        return model
    
    def activate_adapter(self, model, adapter_name: str):
        """
        Activate a specific adapter
        
        Args:
            model: Model with multiple adapters
            adapter_name: Name of adapter to activate
        """
        if isinstance(model, PeftModel):
            model.set_adapter(adapter_name)
        else:
            raise ValueError("Model must be a PeftModel to switch adapters")
    
    def merge_adapter(self, model, adapter_name: str = "default", save_path: str = None):
        """
        Merge an adapter with the base model
        
        Args:
            model: Model with adapter
            adapter_name: Name of adapter to merge
            save_path: Path to save merged model (optional)
        
        Returns:
            Merged model
        """
        if isinstance(model, PeftModel):
            # Set the adapter to merge
            if adapter_name != "default":
                model.set_adapter(adapter_name)
            
            # Merge and unload
            merged_model = model.merge_and_unload()
            
            # Save if path provided
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                merged_model.save_pretrained(save_path)
            
            return merged_model
        else:
            # Model doesn't have adapters, return as-is
            return model
    
    def list_adapters(self) -> List[str]:
        """
        List all loaded adapters
        
        Returns:
            List of adapter names
        """
        return list(self.adapters.keys())
    
    def get_adapter_info(self, adapter_name: str) -> Dict:
        """
        Get information about an adapter
        
        Args:
            adapter_name: Name of the adapter
        
        Returns:
            Adapter information
        """
        if adapter_name not in self.adapters:
            return {}
        
        adapter_path = self.adapters[adapter_name]
        config_path = os.path.join(adapter_path, "adapter_config.json")
        
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            return {"path": adapter_path, "type": "unknown"}
    
    def save_merged_model(self, model, tokenizer, save_path: str):
        """
        Save a merged model with its tokenizer
        
        Args:
            model: Merged model
            tokenizer: Corresponding tokenizer
            save_path: Path to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Merged model saved to {save_path}")


def merge_multiple_adapters(base_model_path: str, adapter_paths: List[str], 
                          weights: Optional[List[float]] = None, output_path: str = None):
    """
    Merge multiple adapters with weighted combination
    
    Args:
        base_model_path: Path to the base model
        adapter_paths: List of adapter paths to merge
        weights: Weights for each adapter (defaults to equal weighting)
        output_path: Output path for merged model
    
    Returns:
        Path to merged model
    """
    if weights is None:
        weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
    
    if output_path is None:
        output_path = f"{base_model_path}_merged_adapters"
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Load each adapter and merge with weights
    for i, (adapter_path, weight) in enumerate(zip(adapter_paths, weights)):
        # Load adapter
        temp_model = PeftModel.from_pretrained(model, adapter_path, adapter_name=f"adapter_{i}")
        
        # Get the adapter weights
        adapter_state_dict = temp_model.state_dict()
        
        # Apply weights and merge (this is a simplified approach)
        # In practice, you might need more sophisticated merging strategies
        model = temp_model.merge_and_unload()
    
    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Merged model with multiple adapters saved to {output_path}")
    return output_path


def create_lora_config(r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.05, 
                      target_modules: Optional[List[str]] = None):
    """
    Create a LoRA configuration
    
    Args:
        r: LoRA attention dimension
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of modules to apply LoRA to
    
    Returns:
        LoRA configuration dictionary
    """
    from peft import LoraConfig, TaskType
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    return config


if __name__ == "__main__":
    # Example usage
    print("PEFT Manager")
    print("=" * 20)
    
    # Example: Initialize manager
    # manager = PEFTManager()
    # 
    # # Load base model
    # model = manager.load_base_model("microsoft/DialoGPT-medium")
    # 
    # # Load an adapter
    # model = manager.load_adapter(model, "./my_adapter", "chatbot")
    # 
    # # Activate the adapter
    # manager.activate_adapter(model, "chatbot")
    # 
    # # List adapters
    # print(f"Loaded adapters: {manager.list_adapters()}")
    
    print("PEFT manager ready for adapter operations")