"""
Multi-GPU Trainer for Sign Language Translation
Includes Repetition Penalties and Metric Type-Safety Fixes
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, List
from contextlib import nullcontext
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ..utils.metrics import compute_bleu, compute_rouge

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        use_wandb: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.use_wandb = use_wandb and rank == 0
        self.is_main_process = (rank == 0)
        
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self.is_main_process:
             logger.info(f"Using mixed precision dtype: {self.dtype}")

        self.use_amp = config.get("mixed_precision", True)
        self.scaler = GradScaler(enabled=(self.use_amp and self.dtype == torch.float16))
        
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
        self.num_epochs = config.get("num_epochs", 30)
        self.save_every = config.get("save_every", 5)
        self.eval_every = config.get("eval_every", 1)
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if self.use_wandb:
                wandb.init(
                    project=config.get("project_name", "sign-language-translation"),
                    name=config.get("run_name", "experiment"),
                    config=config
                )
                
        self.best_val_bleu = 0.0
        self.setup_scheduler()
        
    def setup_scheduler(self):
        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
            
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch}/{self.num_epochs}",
            disable=not self.is_main_process
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            poses = batch["input_ids"].to(self.device)
            pose_masks = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            context = torch.autocast(device_type="cuda", dtype=self.dtype) if self.use_amp else nullcontext()
            
            with context:
                outputs = self.model(
                    input_ids=poses,
                    attention_mask=pose_masks,
                    labels=labels
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
            loss_val = loss.item() * self.gradient_accumulation_steps
            total_loss += loss_val
            
            if self.is_main_process:
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                if self.use_wandb and step % 10 == 0:
                    wandb.log({
                        "train/loss_step": loss_val,
                        "train/lr_projector": self.optimizer.param_groups[0]['lr'],
                        "train/lr_lora": self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch
                    })
                    
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "val") -> Dict:
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(loader, desc=f"Evaluating {prefix}", disable=not self.is_main_process)
        
        for batch in pbar:
            poses = batch["input_ids"].to(self.device)
            pose_masks = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            context = torch.autocast(device_type="cuda", dtype=self.dtype) if self.use_amp else nullcontext()
            
            with context:
                # Generation with strict loop-breaking penalties
                gen_kwargs = {
                    "input_ids": poses,
                    "attention_mask": pose_masks,
                    "max_length": self.config.get("max_gen_length", 128),
                    "num_beams": self.config.get("num_beams", 1),
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3,
                }
                
                generated = self.model.module.generate(**gen_kwargs) if isinstance(self.model, DDP) else self.model.generate(**gen_kwargs)
            
            if self.world_size > 1:
                max_len = torch.tensor(generated.size(1), device=self.device)
                dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
                
                if generated.size(1) < max_len:
                    pad_tensor = torch.full(
                        (generated.size(0), max_len - generated.size(1)),
                        self.tokenizer.pad_token_id,
                        dtype=generated.dtype,
                        device=self.device
                    )
                    generated = torch.cat([generated, pad_tensor], dim=1)
                
                generated = generated.contiguous()
                labels = labels.contiguous()
                
                gathered_preds = [torch.zeros_like(generated) for _ in range(self.world_size)]
                gathered_labels = [torch.zeros_like(labels) for _ in range(self.world_size)]
                
                dist.all_gather(gathered_preds, generated)
                dist.all_gather(gathered_labels, labels)
                
                generated = torch.cat(gathered_preds, dim=0)
                labels = torch.cat(gathered_labels, dim=0)
            
            if self.is_main_process:
                decoded_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                all_preds.extend(decoded_preds)
                all_labels.extend(decoded_labels)
                
        metrics = {}
        if self.is_main_process:
            all_preds = [p.strip() for p in all_preds]
            all_labels = [l.strip() for l in all_labels]
            
            try:
                metrics["bleu_4"] = compute_bleu(all_preds, all_labels)
                rouge_scores = compute_rouge(all_preds, all_labels)
                metrics.update(rouge_scores)
            except Exception as e:
                logger.error(f"Error computing metrics: {e}")
                metrics = {"bleu_4": 0.0, "rouge_l": 0.0}
            
            if self.use_wandb and len(all_preds) > 0:
                sample_df = pd.DataFrame({
                    "Target": all_labels[:10],
                    "Prediction": all_preds[:10]
                })
                wandb.log({f"{prefix}/samples": wandb.Table(dataframe=sample_df)})
                
        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not self.is_main_process:
            return
            
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        
        state_dict = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_bleu": self.best_val_bleu,
            "config": self.config
        }
        
        torch.save(state_dict, self.checkpoint_dir / "latest.pt")
        
        if epoch % self.save_every == 0:
            torch.save(state_dict, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
            
        if is_best:
            torch.save(state_dict, self.checkpoint_dir / "best_model.pt")
            logger.info(f"Saved new best model with BLEU-4: {self.best_val_bleu:.4f}")

    def train(self):
        logger.info("Starting training...")
        
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            
            if self.is_main_process:
                logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
                if self.use_wandb:
                    wandb.log({"train/loss_epoch": train_loss, "epoch": epoch})
                    
            if epoch % self.eval_every == 0:
                val_metrics = self.evaluate(self.val_loader, prefix="val")
                
                if self.is_main_process:
                    logger.info(f"Validation Metrics: {val_metrics}")
                    if self.use_wandb:
                        # Log nested dictionaries properly in wandb
                        flat_metrics = {}
                        for k, v in val_metrics.items():
                            if isinstance(v, dict):
                                for sub_k, sub_v in v.items():
                                    flat_metrics[f"val/{k}_{sub_k}"] = sub_v
                            else:
                                flat_metrics[f"val/{k}"] = v
                        wandb.log(flat_metrics)
                        
                    # --- CRITICAL FIX: Safe Dictionary Extraction ---
                    current_bleu = val_metrics.get("bleu_4", 0.0)
                    
                    # If the metric returned a dictionary (e.g., {'bleu1': X, 'bleu4': Y})
                    if isinstance(current_bleu, dict):
                        current_bleu = current_bleu.get("bleu4", current_bleu.get("bleu_4", 0.0))
                    
                    # Ensure it's a float before comparing
                    try:
                        current_bleu = float(current_bleu)
                    except (ValueError, TypeError):
                        current_bleu = 0.0
                        
                    is_best = current_bleu > self.best_val_bleu
                    if is_best:
                        self.best_val_bleu = current_bleu
                        
                    self.save_checkpoint(epoch, is_best=is_best)
            
            if self.world_size > 1:
                dist.barrier()
        
        if self.is_main_process:
            logger.info("Training completed!")
            logger.info(f"Best Val BLEU-4: {self.best_val_bleu*100:.2f}")
            if self.use_wandb:
                wandb.finish()


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requested but CUDA is not available.")

        torch.cuda.set_device(local_rank)
        
        dist.init_process_group(
            backend='nccl', 
            timeout=timedelta(seconds=7200)
        )
        
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()