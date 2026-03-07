"""
Model Factory for Sign Language Translation - Universal Edition
Includes CRITICAL fixes for Instruction Prompting, Logit Alignment, and Temporal Motion
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. Install with: pip install peft bitsandbytes")


class FeatureProjection(nn.Module):
    """
    UPGRADED: Temporal Feature Projection
    Uses 1D Convolution to understand motion and velocity between frames,
    rather than just looking at static independent poses.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1D Conv mixes information from adjacent frames (kernel_size=3)
        self.temporal_conv = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1 # Keeps time dimension T identical
        )
        
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Time, 266]
        x = x.transpose(1, 2)  # -> [Batch, 266, Time] for Conv1D
        x = self.temporal_conv(x) # -> [Batch, hidden_dim, Time]
        x = x.transpose(1, 2)  # -> [Batch, Time, hidden_dim]
        return self.net(x)


class SignLanguageTranslationModel(nn.Module):
    """Universal wrapper for Seq2Seq and Causal LM models"""

    def __init__(
        self,
        model: nn.Module,
        feature_projection: nn.Module,
        tokenizer: PreTrainedTokenizer,
        is_encoder_decoder: bool,
    ):
        super().__init__()
        self.model = model
        self.feature_projection = feature_projection
        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        logger.info(f"Model type: {'Seq2Seq' if is_encoder_decoder else 'Causal LM'}")

        # --- CRITICAL FIX 1: STRICT CHAT BOUNDING ---
        # Forcing the model into an instruction-following state to stop textbook hallucinations
        self.prompt_text = "User: Translate the following sign language kinematics into an English sentence.\n\nVideo Data: "
        self.suffix_text = "\n\nAssistant: "
        
        # We add_special_tokens=True for the prefix to get the BOS token, False for suffix
        self.prompt_tokens = self.tokenizer(self.prompt_text, return_tensors="pt", add_special_tokens=True).input_ids
        self.suffix_tokens = self.tokenizer(self.suffix_text, return_tensors="pt", add_special_tokens=False).input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        projected = self.feature_projection(input_ids)
        batch_size = input_ids.size(0)
        device = input_ids.device
        dtype = attention_mask.dtype

        # Prepare Prompt and Suffix Embeddings
        prompt_ids = self.prompt_tokens.to(device).expand(batch_size, -1)
        suffix_ids = self.suffix_tokens.to(device).expand(batch_size, -1)
        
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids)
        suffix_embeds = self.model.get_input_embeddings()(suffix_ids)
        
        prompt_attn = torch.ones(batch_size, prompt_embeds.size(1), device=device, dtype=dtype)
        suffix_attn = torch.ones(batch_size, suffix_embeds.size(1), device=device, dtype=dtype)

        if self.is_encoder_decoder:
            combined_embeds = torch.cat([prompt_embeds, projected, suffix_embeds], dim=1)
            combined_attention = torch.cat([prompt_attn, attention_mask, suffix_attn], dim=1)
            
            if labels is None:
                outputs = self.model(inputs_embeds=combined_embeds, attention_mask=combined_attention, return_dict=True)
                return {"loss": None, "logits": outputs.logits}
                
            outputs = self.model(inputs_embeds=combined_embeds, attention_mask=combined_attention, labels=labels, return_dict=True)
            return {"loss": outputs.loss, "logits": outputs.logits}

        # Causal LM Generation Mode
        if labels is None:
            combined_embeds = torch.cat([prompt_embeds, projected, suffix_embeds], dim=1)
            combined_attention = torch.cat([prompt_attn, attention_mask, suffix_attn], dim=1)
            outputs = self.model(inputs_embeds=combined_embeds, attention_mask=combined_attention, return_dict=True)
            return {"loss": None, "logits": outputs.logits}

        # Causal LM Training Mode
        text_embeds = self.model.get_input_embeddings()(labels)
        text_attention = (labels != self.tokenizer.pad_token_id).to(dtype)

        # Concatenate: [Prompt] -> [Video] -> [Suffix] -> [Target Text]
        combined_embeds = torch.cat([prompt_embeds, projected, suffix_embeds, text_embeds], dim=1)
        combined_attention = torch.cat([prompt_attn, attention_mask, suffix_attn, text_attention], dim=1)

        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            return_dict=True,
        )

        # --- CRITICAL FIX 2: PERFECT LOSS ALIGNMENT ---
        prefix_length = prompt_embeds.size(1) + projected.size(1) + suffix_embeds.size(1)
        
        # Logits predicting the labels start exactly at (prefix_length - 1)
        shift_logits = outputs.logits[:, prefix_length - 1 : -1, :].contiguous()
        shift_labels = labels.contiguous()

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return {"loss": loss, "logits": shift_logits}

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 5,
        **kwargs,
    ):
        projected = self.feature_projection(input_ids)
        batch_size = input_ids.size(0)
        device = input_ids.device
        dtype = attention_mask.dtype
        
        prompt_ids = self.prompt_tokens.to(device).expand(batch_size, -1)
        suffix_ids = self.suffix_tokens.to(device).expand(batch_size, -1)
        
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids)
        suffix_embeds = self.model.get_input_embeddings()(suffix_ids)
        
        prompt_attn = torch.ones(batch_size, prompt_embeds.size(1), device=device, dtype=dtype)
        suffix_attn = torch.ones(batch_size, suffix_embeds.size(1), device=device, dtype=dtype)
        
        combined_embeds = torch.cat([prompt_embeds, projected, suffix_embeds], dim=1)
        combined_attention = torch.cat([prompt_attn, attention_mask, suffix_attn], dim=1)
        
        if self.is_encoder_decoder:
            return self.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                max_length=max_length,
                num_beams=num_beams,
                **kwargs,
            )
            
        return self.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            max_new_tokens=max_length,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )


class ModelFactory:
    @staticmethod
    def create_model(
        model_name: str,
        num_keypoints: int,
        tokenizer: PreTrainedTokenizer,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> SignLanguageTranslationModel:

        logger.info(f"Loading model: {model_name}")

        config = AutoConfig.from_pretrained(model_name)
        is_encoder_decoder = config.is_encoder_decoder

        load_kwargs = kwargs.copy()

        if load_in_8bit or load_in_4bit:
            if not PEFT_AVAILABLE:
                raise ImportError("bitsandbytes + peft required")

            load_kwargs["device_map"] = "auto"
            load_kwargs["load_in_8bit"] = load_in_8bit
            load_kwargs["load_in_4bit"] = load_in_4bit

        if is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        try:
            config_vocab_size = getattr(model.config, "vocab_size", None)
            if config_vocab_size is not None and len(tokenizer) > config_vocab_size:
                model.resize_token_embeddings(len(tokenizer))
            elif config_vocab_size is None:
                model.resize_token_embeddings(len(tokenizer))
        except Exception as e:
            logger.warning(f"Could not resize token embeddings: {e}")

        if hasattr(config, "text_config"):
            hidden_size = getattr(config.text_config, "hidden_size", 2560)
        elif hasattr(config, "hidden_size"):   
            hidden_size = config.hidden_size
        elif hasattr(config, "d_model"):
            hidden_size = config.d_model
        else:
            raise ValueError("Cannot determine hidden size")

        if load_in_8bit or load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        if use_lora:
            lora_cfg = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.SEQ_2_SEQ_LM if is_encoder_decoder else TaskType.CAUSAL_LM,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            }
            if lora_config:
                lora_cfg.update(lora_config)

            model = get_peft_model(model, LoraConfig(**lora_cfg))
            model.print_trainable_parameters()

        elif freeze_encoder and is_encoder_decoder:
            for p in model.get_encoder().parameters():
                p.requires_grad = False

        elif freeze_decoder:
            for p in model.parameters():
                p.requires_grad = False

        feature_projection = FeatureProjection(
            input_dim=num_keypoints,
            output_dim=hidden_size,
            dropout=dropout,
        )

        wrapper = SignLanguageTranslationModel(
            model=model,
            feature_projection=feature_projection,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder,
        )

        total = sum(p.numel() for p in wrapper.parameters())
        trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)

        logger.info(f"Total params: {total:,}")
        logger.info(f"Trainable params: {trainable:,}")

        return wrapper