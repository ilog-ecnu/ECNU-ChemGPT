import os
import json

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

def save_json(file_path,data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f'Save {file_path} is ok!')

dirs=os.listdir('chemgpt')
data=[]
for one in dirs:
    file_path='chemgpt/'+one
    x=read_json_file(file_path)
    data+=[x]
save_path='XX.json'
save_json(save_path,data)