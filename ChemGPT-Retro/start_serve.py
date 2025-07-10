from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from utils import normalize_to_open_interval

app = Flask(__name__)

model_id = 'single_step/t5/saved/2024_07_05-21_25_50k'
device_ids = [0, 1, 2, 3]  # 指定的 GPU ID  [0, 1, 2, 3]

# 初始化模型到每个 GPU
models = {}
for gpu_id in device_ids:
    model = T5ForConditionalGeneration.from_pretrained(model_id).cuda(gpu_id)
    models[gpu_id] = model

tokenizer = T5Tokenizer.from_pretrained(model_id)

@app.route('/', methods=['POST'])
def infer():
    data = request.json
    input_text = data.get('input_text')
    gpu_id = data.get('gpu_id', 1)  # 允许通过请求指定使用哪个 GPU
    beam_size = data.get('beam_size', 10)

    # 选择正确的 GPU 和模型
    model = models[gpu_id]
    device = torch.device(f"cuda:{gpu_id}")
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).input_ids.to(device)
    outputs = model.generate(input_ids=input_ids, return_dict_in_generate=True, output_scores=True, do_sample=False, max_length=256, num_beams=10, num_return_sequences=beam_size,diversity_penalty=1.5,num_beam_groups=10)

    decoded_outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    decoded_scores = outputs.sequences_scores.cpu().tolist()
    decoded_scores = normalize_to_open_interval(decoded_scores)

    return jsonify({'result': decoded_outputs, 'score': decoded_scores})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004)
