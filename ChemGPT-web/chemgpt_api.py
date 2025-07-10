
import os
import time
import json
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union


import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
from utils import process_response, generate_chatglm3, generate_stream_chatglm3

from sse_starlette.sse import EventSourceResponse
from util.brain_main_chem import brain_agent


# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    # role: Literal["user", "assistant", "system", "function"]
    role: Literal["user", "assistant", "system",  "function", "observation"]
    content: str = None
    metadata: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(BaseModel):
    # role: Optional[Literal["user", "assistant", "system"]] = None
    role: Optional[Literal["user", "assistant", "observation"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    functions: Optional[Union[dict, List[dict]]] = None
    # Additional parameters
    repetition_penalty: Optional[float] = 1.0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None

def find_file_name(prefix,user_start_time,history_b):
    con=history_b[0]['content']
    if len(con)>20:
         con=con[:20]
    re_name=prefix+'/'+user_start_time+con+'.json'
    return re_name

def write2memory_data(history,user_start_time):
    #找原本文件
    prefix='echo_api_data'
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    b_path=find_file_name(prefix,user_start_time,history)

    #写brain_data数据
    with open(b_path, 'w', encoding="utf-8") as f:
        new_d={"conversations":history}
        json.dump(new_d, f, indent=4, ensure_ascii=False)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="chatglm3-6b")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    
    print('type(request) ',type(request))
    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        functions=request.functions,
    )

    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:

        # Use the stream mode to read the first few characters, if it is not a function call, direct stram output
        predict_stream_generator = predict_stream(request.model, gen_params)
        output = next(predict_stream_generator)
        # print(output)
        if not contains_custom_function(output):
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")

        # Obtain the result directly at one time and determine whether tools needs to be called.
        logger.debug(f"First result output：\n{output}")

        function_call = None
        if output and request.functions:
            try:
                function_call = process_response(output, use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        # CallFunction
        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)

            """
            In this demo, we did not register any tools.
            You can use the tools that have been implemented in our `tool_using` and implement your own streaming tool implementation here.
            Similar to the following method:
                function_args = json.loads(function_call.arguments)
                tool_response = dispatch_tool(tool_name: str, tool_params: dict)
            """
            tool_response = ""

            if not gen_params.get("messages"):
                gen_params["messages"] = []

            gen_params["messages"].append(ChatMessage(
                role="assistant",
                content=output,
            ))
            gen_params["messages"].append(ChatMessage(
                role="function",
                name=function_call.name,
                content=tool_response,
            ))

            # Streaming output of results after function calls
            generate = predict(request.model, gen_params)
            return EventSourceResponse(generate, media_type="text/event-stream")

        else:
            # Handled to avoid exceptions in the above parsing function process.
            generate = parse_output_text(request.model, output)
            return EventSourceResponse(generate, media_type="text/event-stream")

    # Here is the handling of stream = False
    response = generate_chatglm3(model, tokenizer, gen_params)

    # Remove the first newline character
    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    response["text"] = response["text"].strip()
    usage = UsageInfo()
    function_call, finish_reason = None, "stop"
    if request.functions:
        try:
            function_call = process_response(response["text"], use_tool=True)
        except:
            logger.warning("Failed to parse tool call, maybe the response is not a tool call or have been answered.")

    if isinstance(function_call, dict):
        finish_reason = "function_call"
        function_call = FunctionCallResponse(**function_call)

    message = ChatMessage(
        role="assistant",
        content=response["text"],
        function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
    )

    logger.debug(f"==== message ====\n{message}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=usage)


async def predict(model_id: str, params: dict):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_chatglm3(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode
        # print('previous_text ',previous_text)

        finish_reason = new_response["finish_reason"]
        if len(delta_text) == 0 and finish_reason != "function_call":
            continue

        function_call = None
        if finish_reason == "function_call":
            try:
                function_call = process_response(decoded_unicode, use_tool=True)
            except:
                logger.warning(
                    "Failed to parse tool call, maybe the response is not a tool call or have been answered.")

        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)

        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
            function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
        )

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def predict_stream(model_id, gen_params):
    """
    The function call is compatible with stream mode output.

    The first seven characters are determined.
    If not a function call, the stream output is directly generated.
    Otherwise, the complete character content of the function call is returned.

    :param model_id:
    :param gen_params:
    :return:
    """
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    def predict(im_user,inputs, chatbot, max_length, top_p
            , temperature, history_b, history_chem, history_v, history_edu
            , user_start_time, past_key_values,Number,Model):
    
    # chatbot, history_b, history_chem, history_v, history_edu, user_start_time, past_key_values=brain_agent(inputs,im_user,history_b,history_chem,history_v, history_edu,past_key_values,max_length,top_p,temperature,chatbot,user_start_time)
    # print(history_b)
    # yield chatbot, history_b, history_chem, history_v, history_edu, user_start_time, past_key_values
    for chatbot, history_b, history_chem, history_v, history_edu, user_start_time, past_key_values in brain_agent(inputs,im_user,history_b,history_chem,history_v, history_edu,past_key_values,max_length,top_p,temperature,chatbot,user_start_time,Number,Model):
        yield chatbot, history_b, history_chem, history_v, history_edu, user_start_time, past_key_values


    for new_response in generate_stream_chatglm3(model, tokenizer, gen_params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode
        # When it is not a function call and the character length is> 7,
        # try to judge whether it is a function call according to the special function prefix
        if not is_function_call and len(output) > 0:#and len(output) > 7

            # Determine whether a function is called
            is_function_call = contains_custom_function(output)
            if is_function_call:
                continue

            # Non-function call, direct stream output
            finish_reason = new_response["finish_reason"]

            # Send an empty string first to avoid truncation by subsequent next() operations.
            if not has_send_first_chunk:
                message = DeltaMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=finish_reason
                )
                chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
                # print('result ',"{}".format(chunk.model_dump_json(exclude_unset=True)))
                yield "{}".format(chunk.model_dump_json(exclude_unset=True))

            send_msg = delta_text if has_send_first_chunk else output
            has_send_first_chunk = True
            message = DeltaMessage(
                content=send_msg,
                role="assistant",
                function_call=None,
            )
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=finish_reason
            )
            chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
            # print('result ',"{}".format(chunk.model_dump_json(exclude_unset=True)))
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    history=[]
    for one in gen_params['messages']:
        if one.role!='assistant':
            history+=[{'role':one.role, 'content':one.content}]
        else:
            history+=[{'role':one.role, 'metadata':one.metadata,'content':one.content}] 
    metadata, sub_content = output.split("\n", maxsplit=1)
    sub_content=sub_content.strip()
    history+=[
        {'role': 'observation', 'content': gen_params['timeinfo']},
        {'role': 'assistant', 'metadata': metadata, 'content': sub_content}
        ]
    user_start_time=history[1]['content']
    write2memory_data(history,user_start_time)
    if is_function_call:
        yield output
    else:
        yield '[DONE]'
    


async def parse_output_text(model_id: str, value: str):
    """
    Directly output the text content of value

    :param model_id:
    :param value:
    :return:
    """
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content=value),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def contains_custom_function(value: str) -> bool:
    """
    Determine whether 'function_call' according to a special function prefix.

    For example, the functions defined in "tool_using/tool_register.py" are all "get_xxx" and start with "get_"

    [Note] This is not a rigorous judgment method, only for reference.

    :param value:
    :return:
    """
    return value and 'get_' in value


if __name__ == "__main__":

    uvicorn.run(app, host='0.0.0.0', port=8888, workers=1)
