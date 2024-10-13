
import base64
import requests
import google.generativeai as genai
from PIL import Image
import replicate
import dashscope
from http import HTTPStatus
import os
import ollama


def Qwen(prompt):
    res = ollama.chat(
        model="qwen2.5:latest",
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ]
    )
    print(res['message']['content'] + '\n')
    return res['message']['content']

def Deepseek(prompt):
    res = ollama.chat(
        model="deepseek-llm:latest",
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ]
    )
    print(res['message']['content'] + '\n')
    return res['message']['content']

def Llama(prompt):
    res = ollama.chat(
        model="llama3.1:latest",
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ]
    )
    print(res['message']['content'] + '\n')
    return res['message']['content']

def InternLM(prompt):
    res = ollama.chat(
        model="internlm2:latest",
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ]
    )
    print(res['message']['content'] + '\n')
    return res['message']['content']

def Vicuna(prompt):
    res = ollama.chat(
        model="vicuna",
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ]
    )
    print(res['message']['content'] + '\n')
    return res['message']['content']

if __name__ == "__main__":
    Vicuna("你的模型是叫什么名字")