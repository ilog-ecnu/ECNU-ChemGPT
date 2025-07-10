from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "/path/to/model"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# export CUDA_VISIBLE_DEVICES=0,1,2,3
TEMPLATE= "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

while True:
    # 获取用户输入
    prompt = input("请输入您的问题或命令（输入'exit'退出,输入'clear'清除历史记录）:\nprompt: ")
    if prompt.lower() == 'exit':
        print("感谢使用，再见！")
        break

    if prompt.lower() == 'clear':
        print("清空历史记录，重新开始对话")
        continue
    
    # 构建消息列表
    messages += [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        chat_template=TEMPLATE,
        add_generation_prompt=True
    )
    text=text+'\n<|im_start|>assistant\n'
    print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f'assistant:{response}\n')

    messages+= [
        {"role": "assistant", "content": response},
    ]
