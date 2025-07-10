import os
import torch
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
import uvicorn, json, datetime


def get_model():
    path_m=''
    device = torch.device("cuda")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(path_m, trust_remote_code=True)
    if 'cuda' in DEVICE: 
        model = AutoModelForCausalLM.from_pretrained(path_m, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
    else: 
        model = AutoModel.from_pretrained(path_m, trust_remote_code=True).float().to(device).eval()
    return tokenizer, model

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')

    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history)
                                   

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'+ '", history:"' + repr(history) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    
    tokenizer, model = get_model()
    model = model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8004, workers=1)

