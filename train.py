"""
Main Training Script for Sign Language Translation
Includes Differential Learning Rate Setup & Keyword Argument Fix
"""

import os
import sys
import yaml
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataloaders.sign_dataloader import create_dataloaders
from models.model_factory import ModelFactory
from src.trainers.trainer import Trainer, setup_distributed, cleanup_distributed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str):
    """Main training function"""
    
    # Load config
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main_process = (rank == 0)
    
    if is_main_process:
        logger.info(f"Using device: {device}")
    
    # Init Tokenizer
    model_name = config['model']['tokenizer']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Dataloaders
    data_config = config['data']
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=data_config['train_path'],
        val_path=data_config['val_path'],
        test_path=data_config['test_path'],
        pose_dir=data_config['pose_dir'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        max_frames=data_config.get('max_frames', 300),
        max_length=data_config.get('max_length', 128),
        step_frames=data_config.get('step_frames', 1),
        num_keypoints=data_config.get('num_keypoints', 266)
    )
    
    # Model
    model_config = config['model']
    
    # --- CRITICAL FIX: Safe Kwargs Extraction ---
    # This strips 'name' and 'tokenizer' out of the dictionary before unpacking 
    # to avoid the "multiple values for keyword argument" TypeError.
    model_kwargs = {k: v for k, v in model_config.items() if k not in ['name', 'tokenizer', 'params']}
    if 'params' in model_config:
        model_kwargs.update(model_config['params'])

    model = ModelFactory.create_model(
        model_name=model_config['name'],
        num_keypoints=data_config.get('num_keypoints', 266),
        tokenizer=tokenizer,
        **model_kwargs
    )
    model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if is_main_process:
            logger.info(f"Model wrapped with DDP (world_size={world_size})")
    
    training_config = config['training']
    weight_decay = float(training_config.get("weight_decay", 0.0))
    betas = training_config.get("betas", (0.9, 0.999))
    betas = tuple(map(float, betas))
    
    # --- CRITICAL FIX: DIFFERENTIAL OPTIMIZATION ---
    # Fast learning for random initialized weights, gentle learning for LLM backbone
    projector_params = []
    lora_params = []
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'feature_projection' in n:
                projector_params.append(p)
            else:
                lora_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': projector_params, 'lr': 1e-3},  
        {'params': lora_params, 'lr': float(training_config.get("learning_rate", 5e-5))}        
    ], weight_decay=weight_decay, betas=betas)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        config=training_config,
        device=device,
        rank=rank,
        world_size=world_size,
        use_wandb=training_config.get('use_wandb', True)
    )
    
    # Train
    trainer.train()
    
    # Cleanup
    cleanup_distributed()
    
    if is_main_process:
        logger.info("Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sign Language Translation Model")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)