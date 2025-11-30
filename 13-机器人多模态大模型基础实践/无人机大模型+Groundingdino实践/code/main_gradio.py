import openai
import re
import argparse
from robot_library import *
import math
import numpy as np
import os
import json
import time
import httpx
from openai import OpenAI
import gradio as gr

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="prompts.txt")
parser.add_argument("--sysprompt", type=str, default="system_prompts.txt")
args = parser.parse_args()


print("Initializing ChatGPT...")
client = OpenAI(api_key="your-key", base_url="https://api.deepseek.com", http_client=httpx.Client(verify=False))
with open(args.sysprompt, "r", encoding="utf-8") as f:
    sysprompt = f.read()

chat_history = [
    {
        "role": "system",
        "content": sysprompt
    },
    {
        "role": "user",
        "content": "move 10 units up"
    },
    {
        "role": "assistant",
        "content": """```python
aw.fly_to([aw.get_drone_position()[0], aw.get_drone_position()[1], aw.get_drone_position()[2]+10])
```

This code uses the `fly_to()` function to move the drone to a new position that is 10 units up from the current position. It does this by getting the current position of the drone using `get_drone_position()` and then creating a new list with the same X and Y coordinates, but with the Z coordinate increased by 10. The drone will then fly to this new position using `fly_to()`."""
    }
]


def ask(prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=chat_history,
        temperature=0
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]


print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)


def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None

print(f"Initializing AirSim...")
aw = AirSimWrapper()
print(f"Done.")

with open(args.prompt, "r", encoding="utf-8") as f:
    prompt = f.read()

ask(prompt)
print("Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")


def display_image(img):
    im = cv2.imread("groundingdino.png")
    return im

with gr.Blocks() as demo:
    # gr.Markdown("# 无人机多模态大语言模型")
    gr.Markdown("""
                # 无人机多模态大语言模型

                无人机多模态大语言模型利用了Deepseek大语言模型+Grounding DINO的多模态开域识别功能，形成自我感知到自主行为的闭环。
                1. Grounding DINO为开放识别大模型，可以返回环境中的各种物体的标签以及位置信息。
                2. Deepseek根据Prompt处理用户输入，并自主生成无人机可执行的代码，完成相关任务。
                """)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.ClearButton([msg, chatbot])
            def respond(message, chat_history):
                response = ask(message)
                bot_message = response
                chat_history.append((message, bot_message))
                code = extract_python_code(response)
                if code is not None:
                    print("Please wait while I run the code in AirSim...")
                    exec(extract_python_code(response))
                    print("Done!\n")
                # time.sleep(2)
                return "", chat_history
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
        with gr.Column():
            img = gr.Image("datawhale.jpg")
            image_button = gr.Button("Display")
        image_button.click(display_image, inputs=img, outputs=img)

demo.launch()