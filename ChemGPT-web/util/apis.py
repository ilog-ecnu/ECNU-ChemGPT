import requests
import pymysql
import json
from util.tools import log,deal_compound_result


def brain(inputs, history, role = None):
    
    url_b = "http://0.0.0.0:7999"
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        "prompt": inputs,
        "history": history,
        "role": role if role else 'user'
    }
    res_brain = requests.post(url_b, headers=headers, json=data)
    response_data = res_brain.json()
    brain_data = response_data.get('response', '')
    history_b = response_data.get('history', '')
    log(f"brain_data : {brain_data}", 'INFO')        
    log(f"history_b : {history_b}", 'INFO')

    return res_brain.status_code,brain_data,history_b

def general(inputs, history_l, max_length, top_p, temperature):
    
    url_l = "http://0.0.0.0:6001"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "prompt": inputs,
        "history": history_l,
        "max_length": max_length,
        "top_p": top_p,
        "temperature": temperature,
    }
    
    res_language = requests.post(url_l, headers=headers, json=data)
    
    log(f"language is done", 'EVENT')
    assert res_language.status_code==200
    response_data = res_language.json()
    response_text = response_data.get('response', '')
    history_l = response_data.get('history', '')

    return response_data,res_language.status_code,response_text,history_l


def EduChat(query,history_edu):
    assert isinstance(query, str)
    history_edu+=[{"role": "user", "content": query}]
    data = {
        "messages": history_edu,
        "functionUsed": ""
    }
    url = 'http://0.0.0.0:5000/chat'

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization,true',
        'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE,OPTIONS',
        'Access-Control-Allow-Origin': '*',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'Origin': 'http://0.0.0.0:7777',
        'Referer': 'http://0.0.0.0:7777/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    }
    response = requests.post(url, json=data, headers=headers, verify=False)  # 注意这里使用了 verify=False 来禁用 SSL 验证，根据需要进行调整
    
    tt=json.loads(response.text)
    history_edu+=[{"role": "assistant", "content": tt['response']}]
    return tt['response'],history_edu

def rag_web(query):
    assert isinstance(query, str)
    data = {
        "messages": [
            {"role": "system", "content": "请问有什么可以帮助您的吗？"},
            {"role": "user", "content": query}
        ],
        "functionUsed": "retrievalQA"
    }
    url = 'http://0.0.0.0:5000/chat'

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization,true',
        'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE,OPTIONS',
        'Access-Control-Allow-Origin': '*',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'Origin': 'http://0.0.0.0:7777',
        'Referer': 'http://0.0.0.0:7777/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    }
    response = requests.post(url, json=data, headers=headers, verify=False)  # 注意这里使用了 verify=False 来禁用 SSL 验证，根据需要进行调整
    print(response)
    tt=json.loads(response.text)
    return tt['response']

def general_chemgpt_QA_stream(inputs,history_l):
    url = 'http://0.0.0.0:6000/v1/chat/completions'

    history_l+=[{"role": "user", "content": inputs},]
    # 要发送的数据
    data = {
        "model": "chemgpt",
        "messages": history_l,
    }

    # 设置请求头
    headers = {
        'Content-Type': 'application/json'
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(data))

    headers = {"User-Agent": "vLLM Client"}

    pload = {
        "model": "chemgpt",
        "stream": True,
        "messages": history_l
    }
    response = requests.post(url,
                             headers=headers,
                             json=pload,
                             stream=True)
    assistant_reply=''
    for chunk in response.iter_lines(chunk_size=1,
                                     decode_unicode=False,
                                     delimiter=b"\n"):
        if chunk:

            # 假设这是你要解析的字符串
            string_data = chunk.decode("utf-8")

            # 然后解析 JSON
            try:
                json_data = json.loads(string_data[6:])  # 从索引6开始，跳过 "data: "
                # 获取 delta 对应的内容
                delta_content = json_data["choices"][0]["delta"]["content"]
                assistant_reply+=delta_content
                yield delta_content
            except KeyError as e:
                delta_content = json_data["choices"][0]["delta"]["role"]
            except json.JSONDecodeError as e:
                history_l+=[{
                        "role": "assistant",
                        "content": assistant_reply,
                        "tool_calls": []
                    },]
                delta_content='[DONE]'
                assert '[DONE]'==chunk.decode("utf-8")[6:]

def generate_image(key_word):
    url = 'http://0.0.0.0:6002/generate_image'
            
    data = {
        "prompt": key_word
    }
    
    response = requests.post(url, json=data)
    log(f"generate image is done", 'EVENT')

    return response

def reverse_compound(key_word,beam_size,Model):
    if Model == 'ChemGPT':
        url = 'http://localhost:9007'
    else:
        url = 'http://localhost:9008'
    
    Reaction_type='<No_React_Type>'
    
    data = {'input_smile': key_word,
            'dataset': 'USPTO_50K',
            'model': Model,
            'beam_size': beam_size,
            'Reaction_type': Reaction_type
            }
    response = requests.post(url, json=data)
    log(f"reverse_compound is done", 'EVENT')
    response_c=response.json()
    if response_c['topk'][0].startswith('RX'):
        response_c['topk']=[x.split(',')[1] for x in response_c['topk']]
    compounds=deal_compound_result(response_c)
    return compounds,response_c['topk']

def vqa_api(im_user,path,key_word,history_v):
    url = 'http://0.0.0.0:8008'

    #save image of user
    
    if im_user is not None:
        im_user.save(path)
    
    headers = {
        'Content-Type': 'application/json'
    }

    data = {"image": path, 'prompt':key_word, 'history_v': history_v}
    
    response = requests.post(url, headers=headers, json=data)
    log(f"chat image is done", 'EVENT')
    return response
