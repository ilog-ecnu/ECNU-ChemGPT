import pytz
import csv
import json
import os
import datetime, sys
from rich.console import Console
from rdkit import Chem
from rdkit.Chem import Draw

console = Console()

def log(event:str, type:str):
	back_frame = sys._getframe().f_back
	if back_frame is not None:
		back_filename = os.path.basename(back_frame.f_code.co_filename)
		back_funcname = back_frame.f_code.co_name
		back_lineno = back_frame.f_lineno
	else:
		back_filename = "Unknown"
		back_funcname = "Unknown"
		back_lineno = "Unknown"
	now = datetime.datetime.now()
	time = now.strftime("%Y-%m-%d %H:%M:%S")
	logger = f"[{time}] <{back_filename}:{back_lineno}> <{back_funcname}()> {type}: {event}"
	if type.lower() == "info":
		style = "green"
	elif type.lower() == "error":
		style = "red"
	elif type.lower() == "critical":
		style = "bold red"
	elif type.lower() == "event":
		style = "#ffab70"
	else:
		style = ""
	console.print(logger, style = style)
	with open('latest.log','a', encoding='utf-8') as f:
		f.write(f'{logger}\n')

def find_file_name(prefix,user_start_time,history_b):
    con=history_b[0]['content']
    if len(con)>20:
         con=con[:20]
    re_name=prefix+'/'+user_start_time+con+'.json'
    return re_name
    # if os.exist.path(re_name):
    #     return 
    # print(history_b)
    # return prefix+'/'+'temp.json'

def write2memory_data(user_time,chat_record,history_b,user_start_time):
    #找原本文件
    prefix='memory_data/chemgpt-2.0'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
        print(f"目录 {prefix} 已创建")

    b_path=find_file_name(prefix,user_start_time,history_b)
    #写brain_data数据
    with open(b_path, 'w', newline='') as f:
        new_d={"conversations":history_b}
        json.dump(new_d, f, indent=2, ensure_ascii=False)
    
def run_codes(Codes):
    import sys
    from io import StringIO

    # 保存原始的标准输出
    original_stdout = sys.stdout

    try:
        # 创建一个StringIO对象来捕获输出
        capture_output = StringIO()
        sys.stdout = capture_output

        # 执行代码
        exec(Codes)

        # 获取捕获的输出
        captured_output = capture_output.getvalue()
    finally:
        # 恢复原始的标准输出
        sys.stdout = original_stdout
    return captured_output

def deal_lang_history(response_data,inputs):
    language_data = response_data.get('response', '')
    temp_his_l = response_data.get('history', '')
    #让语言模型的历史记录里没有记录到网页信息，网页信息比较多
    if len(temp_his_l)==2:
        history_l=[ {'role': 'user', 'content': inputs}, {'role': 'assistant', 'metadata': '', 'content': language_data}]
    else:
        history_l=temp_his_l[:-2]+[ {'role': 'user', 'content': inputs}, {'role': 'assistant', 'metadata': '', 'content': language_data}]
    return language_data,history_l

def get_cur_time(format_t='%Y_%m_%d__%H_%M_%S'):
    from datetime import datetime
    # 获得当前时间
    beijing_timezone = pytz.timezone('Asia/Shanghai')                
    # 获取当前时间
    current_utc_time = datetime.utcnow()                
    # 将当前时间转换为北京时间
    current_beijing_time = current_utc_time.replace(tzinfo=pytz.utc).astimezone(beijing_timezone)
    # 格式化输出北京时间
    return current_beijing_time.strftime(format_t)


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def get_time(brain_str):
    if brain_str=='几点':
        formatted_time = get_cur_time('%H:%M')                 
        # prompt_time="将下面句子翻译成中文：Now the time is "+formatted_time
    elif brain_str=='几号':
        formatted_time = get_cur_time('%Y:%m:%d')        #'%Y年%m月%d日%H时%M分%S秒'         
        # prompt_time="将下面句子翻译成中文：Today is "+formatted_time
    else:
        formatted_time = get_cur_time('%Y:%m:%d:%H:%M')        #'%Y年%m月%d日%H时%M分%S秒'         
        # prompt_time="将下面句子翻译成中文：Sorry, I just know the current time is "+formatted_time
    return formatted_time

def get_wea_prompt(q_time,wea_res,inputs):
    if q_time=='现在' or '今天':               
        wea_prompt="请你根据下面的信息回答用户的问题，用户的输入是： "+inputs+"\n你知道的天气信息是："+wea_res
    elif q_time=='明天':
        wea_prompt="请你根据下面的信息回答用户的问题，用户的输入是： "+inputs+"\n你知道的明天的天气信息是："+wea_res
    else:       
        wea_prompt="将下面句子翻译成中文：Sorry, at present, I can only get the weather information of the designated city, and it is now or tomorrow."
    return wea_prompt

def deal_mol_input(inputs):
    if '<RX' in inputs:
        inputs=inputs.replace('<RX','< RX')

    inputs='<p>'+inputs+'</p>'
    return inputs

def deal_compound_result(response_c):
    res_compound=response_c.get('topk','')
    res_score=response_c.get('score','')
    return_str=f'''<table border="1">
    <tr>
        <th>编号</th>
        <th>得分</th>
        <th>潜在反应物</th>
    </tr>'''#</table>\n<table border="1">

    for idx,x in enumerate(res_compound):
        # x="<div style='max-width:100%; max-height:360px; overflow:auto'>"  + mdtex2html.convert(parse_text(x))  + "</div>"
        tt=f'''<tr>
        <td>{str(idx+1)}</td>
        <td>{res_score[idx]}</td>
        <td>{x}</td>
    </tr>
'''
        return_str+=tt
    return_str+="</table>\n"
    # log(f"compound return_str:{return_str}", 'INFO')

    return return_str
    
def draw_mol(smiles,time_info,number):

    output_dir = 'path/'+ "num_{}_{}.png".format(time_info,number)

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print('当前SMILES无效')
    else:
        canonical_mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(canonical_mol, output_dir , size=(600, 600), imageType='png', dpi=1200)  # 高清参数
    return output_dir

if __name__=="__main__":
    pass


    