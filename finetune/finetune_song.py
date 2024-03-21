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

base_tones = {
    'C' : 0, 'C#': 1, 'D' : 2, 'D#': 3,
    'E' : 4, 'F' : 5, 'F#': 6, 'G' : 7,
    
    'G#': 8, 'A' : 9, 'A#':10, 'B' :11,
}
line_index = {
    0: 'first', 1 : 'second', 2: 'third',
    3 : 'fourth', 4 : 'fifth', 
    5: 'sixth', 6 : 'seventh',
    7: 'eighth', 8 : 'ninth', 9: 'tenth',
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/petrelfs/share_data/dongxiaoyi/share_models/new7B_SFT")

meta_instruction = """<|System|>:You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.<eosys>
 """

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


def log_discretize(x, bins=512):
    eps = 1
    x_min = np.log(eps-0.3)
    x_max = np.log(6+eps)
    x = min(6, x)
    x = max(-0.3, x)
    x = np.log(x+eps)
    x = (x-x_min) / (x_max-x_min) * (bins-1)
    return np.round(x).astype(int)

def reverse_log_float(x, bins=512):
    if x == 79: #特判
        return 0
    eps = 1
    x_min = np.log(eps-0.3)
    x_max = np.log(6+eps)
    x = x * (x_max - x_min)/(bins-1) + x_min
    x = np.exp(x) - eps
    return float("{:.3f}".format(x))

def bin_time(list_d):
    bin_list = []
    # isinstance(notes_d, list):
    # duration = list_string.split(' ')
    for item in list_d:
        if not isinstance(item, str):
            item = str(item)
        item_tuple = item.split(' ')
        out = ''
        for item_str in item_tuple:
            item_num = float(item_str)
            # out += f'<{item_num}>'
            bin = log_discretize(item_num)
            out += f'<{bin}>'
        bin_list.append(out)
    return bin_list


def append_song_token(model, tokenizer, config):
    old_token_len = len(tokenizer)
    new_tokens = ['<bol>','<bom>','<bop>','<eol>','<eom>','<eop>']
    for note in base_tones:
        for i in range(-1, 10): # -1 -> 9
            new_tokens.append(f'<{note}{i}>') 
    for t_bin in range(512):
        new_tokens.append(f'<{t_bin}>')
    new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
    new_tokens = list(new_tokens)
    new_tokens.sort()
    tokenizer.add_tokens(new_tokens)
    new_token_len = len(tokenizer)
    model.tokenizer = tokenizer

    weight = nn.Parameter(torch.empty((new_token_len-old_token_len, config.hidden_size)))
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    model.config.vocab_size = new_token_len
    model.output.weight.data = torch.cat([model.output.weight, weight.to(model.device)], dim=0)
    model.output.weight.requires_grad = True

    new_token_embed = torch.randn(new_token_len-old_token_len, config.hidden_size)
    new_weight = torch.cat([model.model.tok_embeddings.weight, new_token_embed.to(model.device)], dim=0)
    model.model.vocab_size = new_token_len
    model.model.tok_embeddings.weight.data = new_weight
    model.model.tok_embeddings.weight.requires_grad = True
    
    return model, tokenizer


def generate_lyrics(raw_lyric):
    ##################pure lyrics####################
    line_tuple = []
    line_list = []
    for lyric in raw_lyric:
        for l in lyric:
            line_input = f'{l}'
            line_tuple.append(line_input)
        line_list.append('|'.join(line_tuple))
        line_tuple = []
    pure_lyric_input = ''
    for i, item in enumerate(line_list):
        pure_lyric_input += f" The {line_index[i]} line:" + line_list[i] + '\n'
    ######todo: chinese and english
    # pure_lyric_input = f'The following is pure lyrics. Total {i+1} lines.' + pure_lyric_input
    pure_lyric_input = f'<sol> Total {i+1} lines.' + pure_lyric_input + '<eol>'
    return pure_lyric_input + '[UNUSED_TOKEN_145]'

def note_shift(note, shift_digit):
    note_list = note.split(' ')
    result_list = []
    for note_single in note_list:
        if '<' in note_single:
            now = pretty_midi.note_name_to_number(note_single[1:-1]) + shift_digit
        else:
            now = pretty_midi.note_name_to_number(note_single) + shift_digit
        result_list.append(f'<{pretty_midi.note_number_to_name(now)}>')
    return ' '.join(result_list)

def generate_melody(notes, notes_d, rest_d, shift_digit):
    ##################pure melody####################
    num_line = len(notes)
    # pure_melody_input = f'The following is pure melody. Total {num_line} lines.'
    pure_melody_input = f'<som> Total {num_line} lines.'
    line_tuple = []
    line_list = [] 
    for note, note_d, r_d in zip(notes, notes_d, rest_d):
        for n, n_d, r_d_ in zip(note, bin_time(note_d), bin_time(r_d)):
            line_input = f'{note_shift(n, shift_digit-4)},{n_d},{r_d_}'
            line_tuple.append(line_input)
        line_list.append('|'.join(line_tuple))
        line_tuple = []
    for i, item in enumerate(line_list):
        pure_melody_input += f" The {line_index[i]} line:" + line_list[i] + '\n'
    return pure_melody_input + '<eom>' + '[UNUSED_TOKEN_145]'

def generate_pair(lyrics, notes, notes_d, rest_d):    
    ##################lyric-melody pair####################
    # i_song = random.randint(0, self.num_song-1)
    num_lyrics = len(lyrics)

    line_list, lyric_list, melody_list = [], [], []
    line_tuple, lyric_tuple, melody_tuple = [], [], []
    lyric_input = f'<sol> Total {num_lyrics} lines.'
    melody_input = f'<som> Total {num_lyrics} lines.'
    # text_input = f'The following is lyric-melody pair. Total {num_lyrics} lines.'
    text_input = ''
    for lyric, note, note_d, r_d in zip(lyrics, notes, notes_d, rest_d):
        for l, n, n_d, r_d_ in zip(lyric, note, bin_time(note_d), bin_time(r_d)):
            line_input = f'{l},{n},{n_d},{r_d_}'
            line_tuple.append(line_input)
            lyric_tuple.append(f'{l}')
            melody_tuple.append(f'{n},{n_d},{r_d_}')
        line_list.append('|'.join(line_tuple))
        lyric_list.append('|'.join(lyric_tuple))
        melody_list.append('|'.join(melody_tuple))
        line_tuple = []
        lyric_tuple = []
        melody_tuple = []

    for i, item in enumerate(line_list):
        text_input += f" The {line_index[i]} line:" + line_list[i] + '\n'
        lyric_input += f" The {line_index[i]} line:" + lyric_list[i] + '\n'
        melody_input += f" The {line_index[i]} line:" + melody_list[i] + '\n'
    text_input = f'<sop> Total {i + 1} lines.' + text_input + '<eop>[UNUSED_TOKEN_145]'
    return text_input



class LazyPretrainDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset_kind, use_meta=False, img_size=224):
        super(LazySupervisedDataset, self).__init__()
        self.pure_melody = {'notes': [], 'notes_duration': [], 'rest_duration': []}
        path = f'/mnt/petrelfs/dingshuangrui/MiniGPT-4/corpus/segment/pure_melody_data_segment_{i}.json'
        file = open(path, 'r', encoding='utf-8')
        data = json.load(file)
        self.pure_melody['notes'].extend(data['notes'])
        self.pure_melody['notes_duration'].extend(data['notes_duration'])
        self.pure_melody['rest_duration'].extend(data['rest_duration'])

        self.pure_lyric = []
        path = f'/mnt/petrelfs/dingshuangrui/MiniGPT-4/corpus/segment/pure_lyric_data_segment.json'
        file = open(path, 'r', encoding='utf-8')
        self.pure_lyric.extend(json.load(file))
        
        self.pair_data = {'lyrics': [], 'notes': [], 'notes_duration': [], 'rest_duration': []}
        for i in range(3):
            path = f'/mnt/petrelfs/dingshuangrui/MiniGPT-4/corpus/segment/pair_data_segment_{i}.json'
            file = open(path, 'r', encoding='utf-8')
            data = json.load(file)
            self.pair_data['lyrics'].extend(data['lyrics'])
            self.pair_data['notes'].extend(data['notes'])
            self.pair_data['notes_duration'].extend(data['notes_duration'])
            self.pair_data['rest_duration'].extend(data['rest_duration'])
        
        self.pure_melody['notes'].extend(self.pair_data['notes'])
        self.pure_melody['notes_duration'].extend(self.pair_data['notes_duration'])
        self.pure_melody['rest_duration'].extend(self.pair_data['rest_duration'])
        self.pure_lyric.extend(self.pair_data['lyrics'])     
        assert len(self.pure_melody['notes']) == len(self.pure_melody['notes_duration']) == len(self.pure_melody['rest_duration'])


        if 'pair' in dataset_kind:   
            self.count = len(self.pair_data['lyrics']) * 3
            self.type = 'pair'
        else:
            self.count = len(self.pure_melody['notes']) * 9 + len(self.pure_lyric) 
            self.type = 'pure'
        self.use_meta = use_meta

    def __len__(self):
        return self.count
        # return 100

    def __getitem__(self, i):
        if self.type == 'pair':
            if i // len(self.pair_data['lyrics']) == 0:
                lyrics = self.pair_data['lyrics'][i]
                notes = self.pair_data['notes'][i]
                notes_d = self.pair_data['notes_duration'][i]
                rest_d = self.pair_data['rest_duration'][i]
                conv_text = generate_pair(lyrics, notes, notes_d, rest_d)
            elif i // len(self.pair_data['lyrics']) == 1:
                i_ = i % len(self.pair_data['lyrics'])
                step = len(self.pure_lyric) //  len(self.pair_data['lyrics'])
                i__ = min(int(i_*step), len(self.pure_lyric)-1)
                conv_text = generate_lyrics(self.pure_lyric[int(i__)])
                # print(i__)
            else:
                i_ = i % len(self.pair_data['lyrics'])
                step = len(self.pure_melody['notes']) //  len(self.pair_data['lyrics'])
                i__ = min(int(i_*step), len(self.pure_melody['notes'])-1)
                conv_text = generate_melody(self.pure_melody['notes'][i__],
                                            self.pure_melody['notes_duration'][i__],
                                            self.pure_melody['rest_duration'][i__],
                                            4
                                            )
                
        else:
            if i < len(self.pure_lyric):
                conv_text = generate_lyrics(self.pure_lyric[i])
            else:
                i_ = (i-len(self.pure_lyric)) // 9
                shift = (i-len(self.pure_lyric)) % 9
                conv_text = generate_melody(self.pure_melody['notes'][i_],
                                            self.pure_melody['notes_duration'][i_],
                                            self.pure_melody['rest_duration'][i_],
                                            shift
                                            )
        sample = dict(
            text_input = conv_text,
        )
        
        sample['data_type'] = 'text'

        return dict(
            samples = sample
        )


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

def train_data(training_args, parser, model, tokenizer, data_type, is_save):
    
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
        tokenizer=tokenizer, data_args=data_args, data_type=data_type,
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
    tokenizer: transformers.PreTrainedTokenizer, data_args, data_type='pair',
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print("Loading data...")
    

    
    if data_type == 'sft':
        train_dataset = LazySftDataset()
    else:
        # train_dataset = OnlypairDataset()
        train_dataset = LazySupervisedDataset(data_type, data_args.use_meta, data_args.img_size)
 
    eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
               ), len(train_dataset)


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
    # import pdb; pdb.set_trace()
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

    
    # model, tokenizer = append_song_token(model, tokenizer, config)
    model.tokenizer = tokenizer
    # model = train_data(training_args, parser, model, tokenizer, 'pure', False)
    # model = train_data(training_args, parser, model, tokenizer, 'pair', True)
    model = train_data(training_args, parser, model, tokenizer, 'sft', True)

    

if __name__ == "__main__":
    train()
