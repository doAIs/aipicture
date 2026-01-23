"""
Text-to-Image Trainer - Fine-tune text-to-image models (e.g., Stable Diffusion)
"""
import os
import sys
import torch
import numpy as np
from typing import Dict, Optional, Callable, Union
from pathlib import Path
import json
from PIL import Image
import hashlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


class TextToImageTrainer:
    """
    Trainer for text-to-image models using Hugging Face Diffusers
    Supports fine-tuning of Stable Diffusion models with various techniques:
    - Full fine-tuning
    - LoRA (Low-Rank Adaptation)
    - DreamBooth
    """
    
    def __init__(self, base_model: str = "runwayml/stable-diffusion-v1-5", 
                 dataset_path: str = None, output_dir: str = "./text_to_image_output",
                 use_lora: bool = False, lora_rank: int = 4):
        """
        Initialize the text-to-image trainer
        
        Args:
            base_model: Pre-trained model name or path
            dataset_path: Path to the dataset directory
            output_dir: Output directory for trained model
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_rank: Rank for LoRA layers (lower = fewer parameters)
        """
        self.base_model = base_model
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.device = get_device()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model components
        self.pipeline = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.vae = None
        
        # Initialize training components
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.weight_dtype = torch.float32
        
    def initialize_model(self):
        """
        Initialize the diffusion model and its components
        """
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from peft import LoraConfig, get_peft_model
        
        # Load the models
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae
        
        # Move to device
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        # Set weight dtype for mixed precision training
        if self.device == "cuda":
            self.weight_dtype = torch.float16
            self.unet.to(torch_dtype=self.weight_dtype)
            self.text_encoder.to(torch_dtype=self.weight_dtype)
            self.vae.to(torch_dtype=self.weight_dtype)
        
        # Apply LoRA if requested
        if self.use_lora:
            self.apply_lora()
    
    def apply_lora(self):
        """
        Apply Low-Rank Adaptation (LoRA) to the UNet and text encoder
        """
        from peft import LoraConfig, get_peft_model
        
        # LoRA config for UNet
        unet_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            target_modules=["to_q", "to_v", "to_k", "to_out.0"],
            init_lora_weights="gaussian",
            lora_dropout=0.0
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, unet_lora_config)
        
        # Optionally apply LoRA to text encoder as well
        text_encoder_lora_config = LoraConfig(
            r=self.lora_rank // 2,  # Use smaller rank for text encoder
            lora_alpha=self.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            init_lora_weights="gaussian",
            lora_dropout=0.0
        )
        
        self.text_encoder = get_peft_model(self.text_encoder, text_encoder_lora_config)
        
        print(f"Applied LoRA with rank {self.lora_rank} to UNet and text encoder")
    
    def load_dataset(self, dataset_path: str = None):
        """
        Load dataset for training
        Expected format: CSV with 'prompt' and 'image_path' columns, or directory with paired files
        
        Args:
            dataset_path: Path to the dataset (optional, uses self.dataset_path if not provided)
        
        Returns:
            Dataset object compatible with PyTorch DataLoader
        """
        if dataset_path:
            self.dataset_path = dataset_path
        
        if not self.dataset_path:
            raise ValueError("Dataset path must be provided")
        
        # Import required packages
        from torch.utils.data import Dataset, DataLoader
        import pandas as pd
        
        # Check if dataset is in CSV format
        csv_path = os.path.join(self.dataset_path, "data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'prompt' not in df.columns or 'image_path' not in df.columns:
                raise ValueError("CSV must contain 'prompt' and 'image_path' columns")
            
            # Create dataset class
            class TextImageDataset(Dataset):
                def __init__(self, dataframe, tokenizer, vae, text_encoder, max_length=77):
                    self.dataframe = dataframe
                    self.tokenizer = tokenizer
                    self.vae = vae
                    self.text_encoder = text_encoder
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.dataframe)
                
                def __getitem__(self, idx):
                    row = self.dataframe.iloc[idx]
                    image_path = row['image_path']
                    prompt = row['prompt']
                    
                    # Load and preprocess image
                    image = Image.open(image_path).convert("RGB")
                    image = image.resize((512, 512))  # Standard SD resolution
                    image = np.array(image).astype(np.float32) / 255.0
                    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
                    image = torch.tensor(image).permute(2, 0, 1)  # CHW format
                    
                    # Tokenize prompt
                    text_inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    return {
                        "pixel_values": image,
                        "input_ids": text_inputs.input_ids.squeeze(0),
                        "attention_mask": text_inputs.attention_mask.squeeze(0)
                    }
            
            dataset = TextImageDataset(df, self.tokenizer, self.vae, self.text_encoder)
        else:
            # Assume directory structure with paired image-text files
            # Images in images/ folder, prompts in text/ folder with matching names
            image_dir = os.path.join(self.dataset_path, "images")
            text_dir = os.path.join(self.dataset_path, "prompts")
            
            if not os.path.exists(image_dir):
                raise ValueError(f"Images directory not found: {image_dir}")
            
            class TextImageDataset:
                def __init__(self, image_dir, text_dir, tokenizer, vae, text_encoder, max_length=77):
                    self.image_dir = image_dir
                    self.text_dir = text_dir
                    self.tokenizer = tokenizer
                    self.vae = vae
                    self.text_encoder = text_encoder
                    self.max_length = max_length
                    
                    # Get list of image files
                    self.image_files = [f for f in os.listdir(image_dir) 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                
                def __len__(self):
                    return len(self.image_files)
                
                def __getitem__(self, idx):
                    image_file = self.image_files[idx]
                    base_name = os.path.splitext(image_file)[0]
                    
                    # Find corresponding text file
                    text_file = None
                    for ext in ['.txt', '.prompt', '.caption']:
                        txt_path = os.path.join(self.text_dir, f"{base_name}{ext}")
                        if os.path.exists(txt_path):
                            text_file = txt_path
                            break
                    
                    if text_file is None:
                        # If no text file found, use filename as prompt
                        prompt = base_name.replace('_', ' ').replace('-', ' ')
                    else:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            prompt = f.read().strip()
                    
                    # Load and preprocess image
                    image_path = os.path.join(self.image_dir, image_file)
                    image = Image.open(image_path).convert("RGB")
                    image = image.resize((512, 512))  # Standard SD resolution
                    image = np.array(image).astype(np.float32) / 255.0
                    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
                    image = torch.tensor(image).permute(2, 0, 1)  # CHW format
                    
                    # Tokenize prompt
                    text_inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    return {
                        "pixel_values": image,
                        "input_ids": text_inputs.input_ids.squeeze(0),
                        "attention_mask": text_inputs.attention_mask.squeeze(0)
                    }
            
            dataset = TextImageDataset(image_dir, text_dir, self.tokenizer, self.vae, self.text_encoder)
        
        return dataset
    
    def prepare_training_components(self, learning_rate: float = 1e-4):
        """
        Prepare optimizers, schedulers, and other training components
        
        Args:
            learning_rate: Learning rate for the optimizer
        """
        from torch.optim import AdamW
        
        # Set up optimizer
        if self.use_lora:
            # Only optimize LoRA parameters
            params_to_optimize = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
            if self.text_encoder is not None:
                params_to_optimize += list(filter(lambda p: p.requires_grad, self.text_encoder.parameters()))
        else:
            # Optimize all UNet parameters (and optionally text encoder)
            params_to_optimize = self.unet.parameters()
        
        self.optimizer = AdamW(
            params_to_optimize,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2
        )
        
        # Set up scheduler
        from diffusers.optimization import get_scheduler
        self.lr_scheduler = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=10000  # Will be updated based on actual training steps
        )
    
    def train(self, 
              epochs: int = 10,
              batch_size: int = 1,
              learning_rate: float = 1e-4,
              gradient_accumulation_steps: int = 1,
              max_grad_norm: float = 1.0,
              save_steps: int = 500,
              logging_steps: int = 10,
              seed: int = 42,
              dataloader_num_workers: int = 0,
              resolution: int = 512,
              center_crop: bool = True,
              random_flip: bool = False,
              progress_callback: Optional[Callable] = None):
        """
        Train the text-to-image model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Max gradient norm
            save_steps: Save steps
            logging_steps: Logging steps
            seed: Random seed
            dataloader_num_workers: Number of workers for data loading
            resolution: Resolution for training images
            center_crop: Whether to center crop images
            random_flip: Whether to randomly flip images
            progress_callback: Callback for progress updates
        
        Returns:
            Training result
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize model if not already done
        if self.unet is None:
            self.initialize_model()
        
        # Prepare training components
        self.prepare_training_components(learning_rate)
        
        # Load dataset
        dataset = self.load_dataset()
        
        from torch.utils.data import DataLoader
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers
        )
        
        # Training loop
        global_step = 0
        total_steps = len(self.train_dataloader) * epochs
        
        print(f"Starting training for {epochs} epochs ({total_steps} total steps)...")
        
        self.unet.train()
        if self.text_encoder is not None:
            self.text_encoder.train()
        
        # Initialize loss tracker
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device, dtype=self.weight_dtype)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Encode images to latent space
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215  # Scaling factor for SD
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise schedule
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                encoder_hidden_states = self.text_encoder(input_ids)[0]
                
                # Predict the noise residual
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backpropagate
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Track loss
                epoch_loss += loss.detach().item()
                num_batches += 1
                losses.append(loss.detach().item())
                
                # Log progress
                if global_step % logging_steps == 0:
                    avg_loss = sum(losses[-logging_steps:]) / min(len(losses), logging_steps)
                    print(f"Step {global_step}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                
                # Execute progress callback
                if progress_callback and global_step % 10 == 0:
                    progress = (global_step / total_steps) * 100
                    progress_callback(epoch + 1, global_step, total_steps, loss.item(), {"avg_loss": np.mean(losses[-10:])})
                
                global_step += 1
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save the model
        self.save_model()
        
        # Prepare result
        result = {
            "model_path": self.output_dir,
            "final_avg_loss": np.mean(losses),
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_lora": self.use_lora,
                "lora_rank": self.lora_rank
            },
            "total_steps": total_steps
        }
        
        print(f"Training completed. Final average loss: {result['final_avg_loss']:.4f}")
        
        return result
    
    def save_model(self):
        """
        Save the trained model
        """
        from diffusers import StableDiffusionPipeline
        
        # Create pipeline with trained components
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.pipe.safety_checker,
            feature_extractor=self.pipe.feature_extractor,
        )
        
        # Save pipeline
        pipeline.save_pretrained(self.output_dir)
        
        # Save training config
        config = {
            "base_model": self.base_model,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "training_completed": True
        }
        
        with open(os.path.join(self.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {self.output_dir}")
    
    def generate_sample(self, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
        """
        Generate a sample image with the trained model
        
        Args:
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
        
        Returns:
            Generated image
        """
        from diffusers import StableDiffusionPipeline
        
        # Load the trained pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.output_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Generate image
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        return image


if __name__ == "__main__":
    # Example usage
    print("Text-to-Image Trainer")
    print("=" * 25)
    
    # Example: Initialize trainer
    # trainer = TextToImageTrainer(
    #     base_model="runwayml/stable-diffusion-v1-5",
    #     dataset_path="./my_dataset",
    #     output_dir="./output",
    #     use_lora=True,
    #     lora_rank=8
    # )
    #
    # # Train the model
    # result = trainer.train(
    #     epochs=10,
    #     batch_size=1,
    #     learning_rate=1e-4
    # )
    #
    # print(f"Training completed: {result}")
    
    print("Text-to-image trainer ready for training tasks")