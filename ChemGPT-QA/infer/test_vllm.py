import requests
import json

def chemgpt_stream(inputs,history_l):
    url = 'http://localhost:6001/v1/chat/completions'

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

inputs='写一篇论文详细阐述高锰酸钾，主要是应用方面'
# history_chem=[{"role": "system", "content": "You are a helpful assistant."},]
history_chem=[]
for response_text in chemgpt_stream(inputs,history_chem):
    print(response_text,end='')