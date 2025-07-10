from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

start_time = time.time()

model_id = 'single_step/t5/saved/2024_07_05-21_25_50k'
model = T5ForConditionalGeneration.from_pretrained(model_id, device_map=0)
tokenizer = T5Tokenizer.from_pretrained(model_id)

text = "<RX_5>CC/C=C(/CCO[Si](C)(C)C)O[Si](C)(C)C"
input_ids = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).input_ids.to(model.device)
outputs = model.generate(input_ids=input_ids, do_sample=False, max_length=256, num_beams=10, num_return_sequences=10,diversity_penalty=1.5,num_beam_groups=10)

decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(decoded_outputs)

end_time = time.time()
print('用时：', end_time-start_time)