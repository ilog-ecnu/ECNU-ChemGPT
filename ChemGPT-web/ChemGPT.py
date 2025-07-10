import gradio as gr
from util.brain_main_chem import brain_agent_stream,brain_agent

def predict(im_user,inputs, chatbot, max_length, top_p
            , temperature, history_b, history_chem, history_v, history_edu, history_save
            , user_start_time, past_key_values,Number,Model):
    
    Stream_flag=True
    if chatbot[0][0]=="Stream" and chatbot[0][1]=="False":
        Stream_flag=False
    if Stream_flag:
        for chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values in brain_agent_stream(inputs,im_user,history_b,history_chem,history_v, history_edu,history_save,past_key_values,max_length,top_p,temperature,chatbot,user_start_time,Number,Model):
            yield chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values
    else:
        chatbot, history_b, history_chem, history_v, history_edu,history_save, user_start_time, past_key_values=brain_agent(inputs,im_user,history_b,history_chem,history_v, history_edu,history_save,past_key_values,max_length,top_p,temperature,chatbot,user_start_time,Number,Model)
        yield chatbot, history_b, history_chem, history_v, history_edu, history_save, user_start_time, past_key_values

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], None, None

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChemGPT-2.0</h1>""")
    
    chatbot = gr.Chatbot([('如果你想进行逆合成，你可以提供需要推理的SMILE分子式,并类似下面这样提问：\n推理一下<RX_1>c1ccc(Cn2ccc3ccccc32)cc1的逆合成的反应物', None)],height=500)
    
    with gr.Row():
        user_input = gr.Textbox(show_label=False, placeholder="Shift + Enter 换行, Enter 提交",scale=9)#.style(container=False)
        submitBtn = gr.Button("Submit", variant="primary",scale=1)

    with gr.Row():

        Number=gr.Dropdown(["1", "2", "3","4","5"], label="Number", info="The number of retrodisplacement molecules; the greater the number, the longer the time required for reasoning.",value=['1'],multiselect=False
        )
        Model=gr.Dropdown(["Retro3D", "ChemGPT"], label="Model", info="The models used for retroynthesis reasoning, we provided two models: ChemGPT and Retro3D.",value=['Retro3D'],show_label=True,multiselect=False
        )


    # im = gr.Image(type='pil')
    im=gr.State(None)
    user_start_time=gr.State(None)
    history_b = gr.State([])
    history_save = gr.State([])
    history_chem = gr.State([])
    history_v = gr.State([])
    history_edu = gr.State([{"role": "system", "content": "请问有什么可以帮助您的吗？"},])
    past_key_values = gr.State(None)    

    max_length=gr.State(None)   
    top_p=gr.State(None)
    temperature=gr.State(None)

    user_input.submit(predict, [im,user_input, chatbot, max_length, top_p, temperature, history_b,history_chem, history_v, history_edu,history_save,user_start_time, past_key_values,Number,Model],
                    [chatbot, history_b, history_chem,history_v, history_edu,history_save, user_start_time, past_key_values], show_progress=True)
    user_input.submit(reset_user_input, [], [user_input])
    
    
    submitBtn.click(predict, [im,user_input, chatbot, max_length, top_p, temperature, history_b, history_chem, history_v, history_edu,history_save,user_start_time, past_key_values,Number,Model],
                    [chatbot, history_b,history_chem, history_v, history_edu,history_save, user_start_time, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
 

    # emptyBtn.click(reset_state, outputs=[chatbot, history_b, history_chem, past_key_values], show_progress=True)


def main():
    demo.queue().launch(share=True,server_name="0.0.0.0",server_port=9003)


if __name__ == "__main__":
    main()