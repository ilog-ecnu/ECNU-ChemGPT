import copy
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel,AutoTokenizer
from peft import PeftModel

def load_qwen_from_config(model_name_or_path = "/path/to/model"):

    # Load model
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map='cpu',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    return model,tokenizer


def load_lora_model(model,lora_path):
    # model=PeftModel.from_pretrained(model,lora_path,device='cpu')
    model = PeftModel.from_pretrained(model, lora_path,torch_device='cpu',device='cpu')#.to('cpu')
    return model


if __name__=='__main__':

    model,tokenizer=load_qwen_from_config()

    lora_path = ''

    model=load_lora_model(model,lora_path)

    model = model.merge_and_unload()

    output_path=''

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    

