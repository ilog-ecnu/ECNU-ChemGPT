from PIL import Image
from io import BytesIO
import re
import os
import time
from tools.weather.weather import wea_run
from util.tools import parse_text,write2memory_data,deal_compound_result,log,draw_mol
from util.tools import get_cur_time,get_time,deal_lang_history,run_codes,deal_mol_input
from util.apis import brain,general,generate_image,rag_web,reverse_compound,vqa_api,general_chemgpt_QA_stream]
from util.apis import general_chemgpt_QA_stream


def brain_agent(inputs,im_user,history_b,history_chem,history_v, history_edu,history_save,past_key_values,max_length,top_p,temperature,chatbot,user_start_time,Number,Model):
    
    if user_start_time is None:
        user_start_time=get_cur_time('%Y年%m月%d日%H时%M分%S秒')
    user_time=get_cur_time('%Y年%m月%d日%H时%M分%S秒')
    chatbot.append((parse_text(inputs), ""))

    brain_status_code,brain_data,history_b=brain(inputs,history_b)
    history_save+=history_b[-2:]
    if brain_status_code == 200:
        log(f"Brain请求成功", 'EVENT')
        if history_b[-1]['metadata']=='':
            brain_data=brain_data.replace('BrainGPT','ChemGPT')
            chatbot[-1]= (parse_text(inputs), parse_text(brain_data))

        if history_b[-1]['metadata']=='<|General_problem|>':

            all_reply=''
            for response_text in general_chemgpt_QA_stream(inputs,history_chem):#general_qwen_7b_stream
                all_reply+=response_text
            chatbot[-1]= (parse_text(inputs), parse_text(all_reply))
            history_save+=[{
                "role": "assistant",
                "metadata": "",
                "content": all_reply
                }]
        
        if history_b[-1]['metadata']=='<|Search_web|>':
            web_res=rag_web(inputs)
            chatbot[-1]= (parse_text(inputs), parse_text(web_res))
            history_save+=[{
                "role": "assistant",
                "metadata": "",
                "content": web_res
                }]
            
        if history_b[-1]['metadata']=='<|Generate_image|>':
            match = re.search(r"key word='(.*?)'", history_b[-1]['content'])
            assert match
            brain_status_code,response_first,history_b=brain('ok',history_b, role='observation')
            history_save+=history_b[-2:]
            log(f"response_first : {response_first}", 'INFO')
            prompt_img= match.group(1)
            response=generate_image(prompt_img)
            assert response.status_code == 200
            log(f"图像请求成功！", 'EVENT')
            # Convert the response content to a PIL image
            im_agi = Image.open(BytesIO(response.content))
            chatbot[-1]= (parse_text(inputs), parse_text(response_first))
            path="memory_data/history_pic_agi/"+user_time+prompt_img+".jpg"
            if im_agi is not None:
                im_agi.save(path)
            chatbot = chatbot + [(None,(path,))]
            history_save+=[{
                "role": "assistant",
                "metadata": "",
                "content": path
                }]
            
        
        if history_b[-1]['metadata']=='<|Get_time|>':
            match = re.search(r"key word='(.*?)'", history_b[-1]['content'])
            assert match
            key_word=match.group(1)
            time_info=get_time(key_word)
            brain_status_code,response_text,history_b=brain(time_info,history_b, role='observation')
            history_save+=history_b[-2:]
            assert brain_status_code==200
            chatbot[-1]= (parse_text(inputs), parse_text(response_text))
            log(f"time请求成功！", 'EVENT')
        
        if history_b[-1]['metadata']=='<|Get_weather|>':
            match = re.search(r"city='(.*?)'", history_b[-1]['content'])
            assert match
            city = match.group(1)
            match = re.search(r"time='(.*?)'", history_b[-1]['content'])
            assert match
            time_info = match.group(1)
            wea_res=wea_run(city,time_info)
            brain_status_code,response_text,history_b=brain(wea_res,history_b, role='observation')
            history_save+=history_b[-2:]
            assert brain_status_code==200
            chatbot[-1]= (parse_text(inputs), parse_text(response_text))
            log(f"weather请求成功！", 'EVENT')
        
        if history_b[-1]['metadata']=='<|Get_compound|>':
            if Model is None:
                Model='Retro3D'
            if Number is None:
                Number=1
            print(history_b)
            match = re.search(r"compound='(.*?)'", history_b[-1]['content'])
            assert match
            compound = match.group(1)
            brain_status_code,response_first,history_b=brain('ok',history_b, role='observation')
            history_save+=history_b[-2:]
            log(f"response_first : {response_first}", 'INFO')
            deal_inputs=deal_mol_input(inputs)

            chatbot[-1]= ((deal_inputs), parse_text(response_first))
            
            chatbot = chatbot + [(None,f'正在推理生成{Number[0]}个Smile分子式，并尝试生成分子图，请稍等')]
            if isinstance(Number,list):
                Number=Number[0]
            if isinstance(Model,(list,tuple)):
                Model=Model[0]
            
            compounds,mol_ls=reverse_compound(compound,int(Number),Model)
            chatbot = chatbot + [(None,compounds)]
            
            for idx,mol in enumerate(mol_ls):
                
                time.sleep(0.1)
                path=draw_mol(mol,user_time,idx)
                print(path)
                if os.path.exists(path):
                    intro=f'第{idx+1}个潜在反应物的分子图如下'
                    chatbot = chatbot + [(None,intro)]
                    time.sleep(0.5)
                    chatbot = chatbot + [(None,(path,))]
                    
                else:
                    intro=f'抱歉，我推理出来的第{idx+1}个分子可能是一个无效分子，我无法生成有效的分子图'
                    chatbot = chatbot + [(None,intro)]

        if history_b[-1]['metadata']=='<|Chat_image|>':
            response='不好意思，基于图片的聊天功能暂时没有开启'
            chatbot[-1]= (parse_text(inputs), parse_text(response))

    write2memory_data(user_time,chatbot[-2:],history_b,user_start_time)

    return chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values

def brain_agent_stream(inputs,im_user,history_b,history_chem,history_v, history_edu,history_save, past_key_values,max_length,top_p,temperature,chatbot,user_start_time,Number,Model):
    
    if user_start_time is None:
        user_start_time=get_cur_time('%Y年%m月%d日%H时%M分%S秒')
    user_time=get_cur_time('%Y年%m月%d日%H时%M分%S秒')
    chatbot.append((parse_text(inputs), ""))

    brain_status_code,brain_data,history_b=brain(inputs,history_b)
    history_save+=history_b[-2:]

    if brain_status_code == 200:
        log(f"Brain请求成功", 'EVENT')
        if history_b[-1]['metadata']=='':
            brain_data=brain_data.replace('BrainGPT','ChemGPT')
            chatbot[-1]= (parse_text(inputs), parse_text(brain_data))
            yield chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values

        if history_b[-1]['metadata']=='<|General_problem|>':
            all_reply=''
            for response_text in general_chemgpt_QA_stream(inputs,history_chem):#
                all_reply+=response_text
                chatbot[-1]= (parse_text(inputs), parse_text(all_reply))
                yield chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values
            history_save+=[{
                "role": "assistant",
                "metadata": "",
                "content": all_reply
                }]

        if history_b[-1]['metadata']=='<|Search_web|>':
            web_res=rag_web(inputs)
            chatbot[-1]= (parse_text(inputs), parse_text(web_res))
            history_save+=[{
                "role": "assistant",
                "metadata": "",
                "content": web_res
                }]
            
        if history_b[-1]['metadata']=='<|Generate_image|>':
            match = re.search(r"key word='(.*?)'", history_b[-1]['content'])
            assert match
            brain_status_code,response_first,history_b=brain('ok',history_b, role='observation')
            history_save+=history_b[-2:]
            log(f"response_first : {response_first}", 'INFO')
            prompt_img= match.group(1)
            response=generate_image(prompt_img)
            assert response.status_code == 200
            log(f"图像请求成功！", 'EVENT')
            # Convert the response content to a PIL image
            im_agi = Image.open(BytesIO(response.content))
            chatbot[-1]= (parse_text(inputs), parse_text(response_first))
            path="memory_data/history_pic_agi/"+user_time+prompt_img+".jpg"
            if im_agi is not None:
                im_agi.save(path)
            chatbot = chatbot + [(None,(path,))]
            history_save+=[{
                "role": "assistant",
                "metadata": "",
                "content": path
                }]
            
        
        if history_b[-1]['metadata']=='<|Get_time|>':
            match = re.search(r"key word='(.*?)'", history_b[-1]['content'])
            assert match
            key_word=match.group(1)
            time_info=get_time(key_word)
            brain_status_code,response_text,history_b=brain(time_info,history_b, role='observation')
            history_save+=history_b[-2:]
            assert brain_status_code==200
            chatbot[-1]= (parse_text(inputs), parse_text(response_text))
            log(f"time请求成功！", 'EVENT')
        
        if history_b[-1]['metadata']=='<|Get_weather|>':
            match = re.search(r"city='(.*?)'", history_b[-1]['content'])
            assert match
            city = match.group(1)
            match = re.search(r"time='(.*?)'", history_b[-1]['content'])
            assert match
            time_info = match.group(1)
            wea_res=wea_run(city,time_info)
            brain_status_code,response_text,history_b=brain(wea_res,history_b, role='observation')
            history_save+=history_b[-2:]
            assert brain_status_code==200
            chatbot[-1]= (parse_text(inputs), parse_text(response_text))
            log(f"weather请求成功！", 'EVENT')
        
        if history_b[-1]['metadata']=='<|Get_compound|>':
            if Model is None:
                Model='Retro3D'
            if Number is None:
                Number=1
            print(history_b)
            match = re.search(r"compound='(.*?)'", history_b[-1]['content'])
            assert match
            compound = match.group(1)
            brain_status_code,response_first,history_b=brain('ok',history_b, role='observation')
            history_save+=history_b[-2:]
            log(f"response_first : {response_first}", 'INFO')
            deal_inputs=deal_mol_input(inputs)

            chatbot[-1]= ((deal_inputs), parse_text(response_first))
            yield chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values

            chatbot = chatbot + [(None,f'正在推理生成{Number[0]}个Smile分子式，并尝试生成分子图，请稍等')]
            yield chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values
            if isinstance(Number,list):
                Number=Number[0]
            if isinstance(Model,list):
                Model=Model[0]
            compounds,mol_ls=reverse_compound(compound,int(Number),Model)
            chatbot = chatbot + [(None,compounds)]
            yield chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values

            for idx,mol in enumerate(mol_ls):
                
                time.sleep(0.1)
                path=draw_mol(mol,user_time,idx)
                print(path)
                if os.path.exists(path):
                    intro=f'第{idx+1}个潜在反应物的分子图如下'
                    chatbot = chatbot + [(None,intro)]
                    time.sleep(0.5)
                    chatbot = chatbot + [(None,(path,))]
                    yield chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values
                else:
                    intro=f'抱歉，我推理出来的第{idx+1}个分子可能是一个无效分子，我无法生成有效的分子图'
                    chatbot = chatbot + [(None,intro)]

        if history_b[-1]['metadata']=='<|Chat_image|>':
            response='不好意思，基于图片的聊天功能暂时没有开启'
            chatbot[-1]= (parse_text(inputs), parse_text(response))

    write2memory_data(user_time,chatbot[1:],history_save,user_start_time)

    yield chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values