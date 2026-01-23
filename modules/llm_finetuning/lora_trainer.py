"""
LoRA Trainer - Fine-tune LLMs with Low-Rank Adaptation
"""
import os
import sys
import torch
import json
from typing import Dict, Optional, Callable, List
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
from trl import SFTTrainer
import bitsandbytes as bnb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


class LoRATrainer:
    """
    Trainer for fine-tuning LLMs using LoRA (Low-Rank Adaptation)
    Supports parameter-efficient fine-tuning of large language models
    """
    
    def __init__(self, 
                 base_model: str = "microsoft/DialoGPT-medium",
                 dataset_path: str = None,
                 output_dir: str = "./lora_output",
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 target_modules: Optional[List[str]] = None,
                 use_4bit: bool = False,
                 use_8bit: bool = False,
                 gradient_checkpointing: bool = True):
        """
        Initialize the LoRA trainer
        
        Args:
            base_model: Pre-trained model name or path
            dataset_path: Path to the dataset file or directory
            output_dir: Output directory for trained model
            lora_r: LoRA attention dimension
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of modules to apply LoRA to
            use_4bit: Use 4-bit quantization (QLoRA)
            use_8bit: Use 8-bit quantization
            gradient_checkpointing: Use gradient checkpointing to save memory
        """
        self.base_model = base_model
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.gradient_checkpointing = gradient_checkpointing
        self.device = get_device()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "use_cache": False
        }
        
        if self.use_4bit:
            model_kwargs.update({
                "load_in_4bit": True,
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16
                }
            })
        elif self.use_8bit:
            model_kwargs.update({
                "load_in_8bit": True,
                "device_map": "auto"
            })
        
        self.model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        
        # Prepare model for k-bit training if using quantization
        if self.use_4bit or self.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        self.apply_lora()
    
    def apply_lora(self):
        """Apply Low-Rank Adaptation to the model"""
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, config)
        print(f"Applied LoRA with r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def load_dataset(self, dataset_path: str = None):
        """
        Load dataset for training
        Expected format: JSON file with 'prompt' and 'response' fields, or a HuggingFace dataset
        
        Args:
            dataset_path: Path to the dataset (optional, uses self.dataset_path if not provided)
        
        Returns:
            Dataset object
        """
        if dataset_path:
            self.dataset_path = dataset_path
        
        if not self.dataset_path:
            raise ValueError("Dataset path must be provided")
        
        import pandas as pd
        
        # Check if dataset is in JSON format
        if self.dataset_path.endswith('.json'):
            # Load JSON dataset
            data = pd.read_json(self.dataset_path)
            
            # Format the data for instruction tuning
            formatted_texts = []
            for _, row in data.iterrows():
                if 'instruction' in row and 'output' in row:
                    # Alpaca-style format
                    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['output']}\n\n"
                elif 'prompt' in row and 'response' in row:
                    # Prompt-response format
                    text = f"User: {row['prompt']}\nAssistant: {row['response']}\n"
                elif 'text' in row:
                    # Raw text format
                    text = row['text']
                else:
                    # Fallback: combine all fields
                    text = " ".join([str(v) for v in row.values()])
                
                formatted_texts.append(text)
            
            # Create dataset
            from datasets import Dataset
            dataset = Dataset.from_dict({"text": formatted_texts})
            
        elif self.dataset_path.endswith(('.csv', '.xlsx')):
            # Load from CSV or Excel
            if self.dataset_path.endswith('.csv'):
                data = pd.read_csv(self.dataset_path)
            else:
                data = pd.read_excel(self.dataset_path)
            
            # Format the data
            formatted_texts = []
            for _, row in data.iterrows():
                if 'instruction' in row and 'output' in row:
                    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['output']}\n\n"
                elif 'prompt' in row and 'response' in row:
                    text = f"User: {row['prompt']}\nAssistant: {row['response']}\n"
                else:
                    text = " ".join([str(v) for v in row.values()])
                
                formatted_texts.append(text)
            
            from datasets import Dataset
            dataset = Dataset.from_dict({"text": formatted_texts})
        else:
            # Assume it's a HuggingFace dataset directory
            from datasets import load_dataset
            dataset = load_dataset(self.dataset_path)
        
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize the examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def format_instruction(self, instruction: str, input_text: str = "", output: str = ""):
        """
        Format a single instruction example
        
        Args:
            instruction: The instruction
            input_text: Optional input text
            output: Optional output text
        
        Returns:
            Formatted string
        """
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    def train(self,
              epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              gradient_accumulation_steps: int = 4,
              warmup_steps: int = 100,
              weight_decay: float = 0.01,
              max_grad_norm: float = 1.0,
              logging_steps: int = 10,
              eval_steps: int = 500,
              save_steps: int = 500,
              save_total_limit: int = 3,
              group_by_length: bool = True,
              bf16: bool = False,  # Use bfloat16 if available
              seed: int = 42,
              dataloader_num_workers: int = 0,
              progress_callback: Optional[Callable] = None):
        """
        Train the model with LoRA
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay
            max_grad_norm: Max gradient norm
            logging_steps: Logging steps
            eval_steps: Evaluation steps
            save_steps: Save steps
            save_total_limit: Limit of total saved checkpoints
            group_by_length: Group samples by length for efficient batching
            bf16: Use bfloat16 precision
            seed: Random seed
            dataloader_num_workers: Number of workers for data loading
            progress_callback: Callback for progress updates
        
        Returns:
            Training result
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        import numpy as np
        np.random.seed(seed)
        
        # Load dataset
        raw_dataset = self.load_dataset()
        
        # Tokenize dataset
        if 'train' in raw_dataset:
            train_dataset = raw_dataset['train'].map(self.tokenize_function, batched=True)
        else:
            train_dataset = raw_dataset.map(self.tokenize_function, batched=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            group_by_length=group_by_length,
            bf16=bf16,
            bf16_full_eval=bf16,
            dataloader_pin_memory=True,
            dataloader_num_workers=dataloader_num_workers,
            report_to=None,  # Disable reporting to external services
            push_to_hub=False,
            disable_tqdm=True,  # We'll handle progress manually
            gradient_checkpointing=self.gradient_checkpointing,
            remove_unused_columns=False,
        )
        
        # Define data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Enable gradient checkpointing if specified
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Train the model
        print(f"Starting LoRA training for {epochs} epochs...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        trainer.save_state()
        
        # Save LoRA adapter separately
        self.model.save_pretrained(os.path.join(self.output_dir, "lora_adapter"))
        
        # Save training configuration
        config = {
            "base_model": self.base_model,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "use_4bit": self.use_4bit,
            "use_8bit": self.use_8bit,
            "training_completed": True
        }
        
        with open(os.path.join(self.output_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Prepare result
        result = {
            "model_path": self.output_dir,
            "adapter_path": os.path.join(self.output_dir, "lora_adapter"),
            "train_loss": train_result.training_loss,
            "training_args": training_args.to_dict()
        }
        
        print(f"LoRA training completed. Loss: {train_result.training_loss}")
        
        # Execute callback if provided
        if progress_callback:
            # Calculate total steps for progress
            total_steps = len(train_dataset) // (batch_size * gradient_accumulation_steps) * epochs
            progress_callback(0, total_steps, train_result.training_loss, {})
        
        return result
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.7, 
                 top_p: float = 0.9, do_sample: bool = True):
        """
        Generate text with the trained model
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling or greedy decoding
        
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text


def merge_lora_adapter(adapter_path: str, base_model_path: str = None, output_path: str = None):
    """
    Merge LoRA adapter with base model
    
    Args:
        adapter_path: Path to the LoRA adapter
        base_model_path: Path to the base model (defaults to self.base_model)
        output_path: Output path for merged model (defaults to adapter_path + "_merged")
    
    Returns:
        Path to merged model
    """
    if base_model_path is None:
        base_model_path = adapter_path.replace("/lora_adapter", "")  # Infer from adapter path
    
    if output_path is None:
        output_path = f"{adapter_path}_merged"
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Load LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge the adapter
    model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Merged model saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage
    print("LoRA Trainer")
    print("=" * 20)
    
    # Example: Initialize trainer
    # trainer = LoRATrainer(
    #     base_model="microsoft/DialoGPT-medium",
    #     dataset_path="./my_dataset.json",
    #     output_dir="./output",
    #     lora_r=8,
    #     lora_alpha=32,
    #     use_4bit=True  # For QLoRA
    # )
    #
    # # Train the model
    # result = trainer.train(
    #     epochs=3,
    #     batch_size=4,
    #     learning_rate=2e-4
    # )
    #
    # print(f"Training completed: {result}")
    
    print("LoRA trainer ready for LLM fine-tuning tasks")