# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
from dataclasses import dataclass, field
import json
import math
import logging
import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, deepspeed, get_cosine_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from typing import Dict, Optional, Sequence, List
import pretty_midi
 
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_root: str = field(
        default=None, metadata={"help": "Path to the image of training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    use_meta: bool = False
    img_size: int = 224


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ft_head: bool = True
    use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = False
    label_names: List[str] = field(
        default_factory=lambda: ['samples']
    )
    # lr_scheduler_type: str = field(default="cosine_with_restarts")
    # warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    # num_training_steps: int = field(default=100, metadata={"help": "Total number of training steps to perform."})
    # lr_scheduler_type: str = field(default="cosine")
    # def get_scheduler(self, optimizer):
    #     return get_cosine_schedule_with_warmup(optimizer, self.warmup_steps, self.num_training_steps)
    
  
@dataclass
class LoraArguments:
    lora_r: int = 128
    lora_alpha: int = 160
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['mlp.up_proj', 'mlp.down_proj', 'mlp.gate_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param



# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        # to_return = {k: t for k, t in named_params if "lora_" in k}
        to_return = {k: t for k, t in named_params if t.requires_grad}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def conv2text(sources, use_meta=False):
    END_HUMAN = "\n"
    END_BOT = "[UNUSED_TOKEN_0]\n"
    conversation = meta_instruction if use_meta else '<bos>'
    conversation += f"""[UNUSED_TOKEN_146]user\n{sources['question']}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{sources['answer']}[UNUSED_TOKEN_145]\n"""
    return conversation

class LazySftDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_root, use_meta=False, img_size=224):
        super(LazySftDataset, self).__init__()
        file = open(data_root, 'r', encoding='utf-8')
        self.data = json.load(file)
        self.use_meta = use_meta

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        new_i = i % len(self.data)
        conv_text = conv2text(self.data[new_i])
        sample = dict(
            text_input = conv_text,
        )
        
        sample['data_type'] = 'text'

        return dict(
            samples = sample
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance['samples'] for instance in instances]
        text_input, data_type = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "data_type"))
        
        batch = dict(
            text_input = text_input,
            data_type = data_type,
        )
        

        return dict(
            samples=batch
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print("Loading data...")
    
    train_dataset = LazySftDataset(data_args.data_path)

    eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
               ), len(train_dataset)


def train_data(training_args, parser, model, tokenizer, is_save):
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    data_module, len_dataset = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args,
    )
    # Start trainner
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        **data_module
    )
    trainer.train()
    trainer.save_state()
    if is_save:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)
    return model


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    rank0_print(training_args)
    
    #

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = training_args.max_length

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # model = model.maybe_merge_lora()
    ### current length: 92544
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    

    # Load data
    for name, param in model.named_parameters():
        rank0_print(name, param.requires_grad)

    model.tokenizer = tokenizer
    model = train_data(training_args, parser, model, tokenizer, True)

if __name__ == "__main__":
    train()
