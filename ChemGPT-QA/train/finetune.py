
from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        # default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",'up_proj','down_proj','embed_tokens','lm_head']
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
        to_return = {k: t for k, t in named_params if "lora_" in k}
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

def print_tokens_labels(tokens: List[int], target: List[int], tokenizer):
    # print("Sanity Check >>>>>>>>>>>>>")
    import copy
    temp_tokens=copy.deepcopy(tokens[0].tolist())
    temp_target=copy.deepcopy(target[0].tolist())
    save_name='check_token_target.txt'
    # if os.path.exists(save_name):
    #     os.remove(save_name)
    ff = open(save_name,'a+')
    for t, m in zip(temp_tokens, temp_target):
        if t<0:
            decoded='<Image Data>'
        else:
            decoded = tokenizer.batch_decode([t], skip_special_tokens=False)[0]
        # print("%20s: %6d -> %6d" % (repr(decoded), t, m))
        ff.write("%20s: %6d -> %6d\n" % (repr(decoded), t, m))
    ff.close()
    # print("<<<<<<<<<<<<< Sanity Check")
    assert len(tokens) == len(target), f"length mismatch: {len(tokens)} vs {len(target)}"

def mask_user_targets(input_ids):
    target_batch = []
    for bs in range(input_ids.shape[0]):
        ids = input_ids[bs]
        import copy
        targets = copy.deepcopy(ids)
        im_round=0
        id_im_start=0
        # id_im_end=0
        for i, temp_id in enumerate(ids): 
            if temp_id == 151644:
                if i==len(ids)-1:
                    continue
                if ids[i+1] != 77960:
                    im_round+=1
                    if im_round==2:
                        id_im_start=0
                        targets[id_im_start:i + 1] = -100
                        id_im_start=i
                    elif im_round%2==0:
                        id_im_start=i
                    elif im_round%2==1:
                        targets[id_im_start:i + 3] = -100

        target_batch.append(targets.unsqueeze(0))

    target_batch = torch.cat(target_batch, dim=0)
    return target_batch

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:

    input_ids, targets, attention_masks = [], [], []
    TEMPLATE= "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    for i, source in enumerate(sources):
        # print(source)
        text = tokenizer.apply_chat_template(
                source,
                chat_template=TEMPLATE,
                tokenize=False,
                add_generation_prompt=False,
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
        
        remove_str='''<|im_start|>system
You are a helpful assistant.<|im_end|>'''
        text=text.replace(remove_str,'')
        text=remove_str+text
        # print(text)
        part_tokens=tokenizer(
                text,
                return_tensors='pt',
                padding="max_length",
                truncation=True)
        # print(part_tokens)
        input_id=part_tokens.input_ids
        attention_mask=part_tokens.attention_mask
        # print(input_id.shape)
        target=mask_user_targets(input_id)
        assert len(input_id) == len(target)
        input_ids.append(input_id[:max_len])
        attention_masks.append(attention_mask[:max_len])
        targets.append(target)
        # print_tokens_labels(input_id,target,tokenizer)
        # 1/0


    if len(input_ids) == 1:
        input_ids = input_ids[0]
    else:
        input_ids = torch.tensor(input_ids, dtype=torch.int)
    
    if len(attention_masks) == 1:
        attention_masks = attention_masks[0]
    else:
        attention_masks = torch.tensor(attention_masks, dtype=torch.int)

    if len(targets) == 1:
        targets = targets[0]
    else:
        targets = torch.tensor(targets, dtype=torch.int)
    
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    input_ids.to(device)
    attention_masks.to(device)
    targets.to(device)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_masks, #input_ids.ne(tokenizer.pad_token_id)
    )

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError:
        print(f"File {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # train_json = json.load(open(data_args.data_path, "r"))
    train_json=read_json_file(data_args.data_path)

    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


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

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()


    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load tokenizer and data
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    # tokenizer.pad_token_id = tokenizer.eod_id

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )

    print(model)

    if training_args.use_lora:
        if lora_args.q_lora or is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        # model.print_trainable_parameters()

        # ====== 新增：输出LoRA参数量统计 ======
        if local_rank == 0:
            # 计算可训练参数总量
            trainable_params = 0
            all_params = 0
            for _, param in model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            # 计算LoRA参数比例
            lora_percentage = 100 * trainable_params / all_params
            
            # 格式化输出
            print("\n" + "="*50)
            print(f"LoRA参数统计:")
            print(f"可训练参数总量: {trainable_params:,}")
            print(f"模型总参数量: {all_params:,}")
            print(f"LoRA参数量占比: {lora_percentage:.2f}%")
            print("="*50 + "\n")

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        print(model)


    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()