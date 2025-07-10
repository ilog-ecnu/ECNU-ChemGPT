from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import torch
import os

# 设置环境变量来指定使用GPU 0, 1, 2, 3
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def read_txt(path):
    with open(path, "r", encoding='utf-8') as f:
        data = f.readlines()
    return data

def write_txt(content, path):
    with open(path, "a", encoding='utf-8') as f:
        f.write(content)

def format_list(data_list, batch_size=10):
    formatted_strings = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        batch_str = '\t'.join(map(str, batch)) + '\n'
        formatted_strings.append(batch_str)
    return ''.join(formatted_strings)

def batch(list, batch_size):
    for i in range(0, len(list), batch_size):
        yield list[i:i + batch_size]

def inference(text_path, model_id, output_file, batch_size):
    # 加载模型
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    # 使用DataParallel自动并行化模型
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()  # 确保模型在GPU上

    tokenizer = T5Tokenizer.from_pretrained(model_id)
    lines = read_txt(text_path)
    total_batches = (len(lines) + batch_size - 1) // batch_size
    results = ''
    progress_bar = tqdm(total=total_batches)
    for batch_lines in batch(lines, batch_size):
        input_ids = tokenizer(batch_lines, return_tensors="pt", padding=True, max_length=256, truncation=True).input_ids.cuda()
        # 通过model.module.generate来调用generate
        outputs = model.module.generate(input_ids=input_ids, max_length=256, num_beams=10, num_return_sequences=10)

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result = format_list(decoded_outputs)
        results += result
        progress_bar.update(1)
    progress_bar.close()
    write_txt(results, output_file)


if __name__=='__main__':
    input_file = 'test_sources'
    model_id = 'saved/2024_07_05-21_25_50k/checkpoint_6000_50k'
    output_path = 'test_result'
    batch_size = 15
    inference(input_file, model_id, output_path, batch_size=batch_size)
