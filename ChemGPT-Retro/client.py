# client.py
import requests

def run_client(data):
    url = 'http://localhost:8004'
    response = requests.post(url, json=data)
    
    if response.status_code == 200:  # 200 表示请求成功
        result = response.json()
    else:
        print('Error:', response.status_code)
    return result

if __name__ == '__main__':
    input_text = "<RX_6>CC1=C(C(=O)O)SC(C2=CC(C#N)=C(OCC(C)C)C=C2)=N1"
    gpu_id = 3
    beam_size = 10
    data = {
        'input_text':input_text,
        'gpu_id':gpu_id,
        'beam_size': beam_size
    }
    run_client(data)