"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any

import aiolimiter
import openai
from openai import AzureOpenAI
from tqdm.asyncio import tqdm_asyncio
import requests

client = AzureOpenAI()

def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")
    response = openai.Completion.create(  # type: ignore
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token],
    )
    answer: str = response["choices"][0]["text"]
    return answer

# this function is changed temporarily for azure api using
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    model = "gpt-4o-2024-05-13"

    response = client.chat.completions.create(  # type: ignore
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token] if stop_token else None,
    )
    answer: str = response.choices[0].message.content
    return answer

def generate_from_4v_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    # if "OPENAI_API_KEY" not in os.environ:
    #     raise ValueError(
    #         "OPENAI_API_KEY environment variable must be set when using OpenAI API."
    #     )
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    # response = openai.ChatCompletion.create(  # type: ignore
    #     model="gpt-4-vision-preview",
    #     messages=messages,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     top_p=top_p,
    #     api_base = "https://api.openai.com/v1"
    # )
    # answer: str = response["choices"][0]["message"]["content"]
    # return answer


    # get api_key from env
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    api_key = os.environ["OPENAI_API_KEY"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        res = response.json()
        answer: str = res["choices"][0]['message']['content']
    except Exception as e:
        print(e)
        answer = ""
    return answer

# NOTE: temporarily add the funcition here; this is for azure only
def generate_from_4o_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    # count and print latency
    start = time.time()
    if "AZURE_OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "AZURE_OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
    openai.api_base =  os.environ["AZURE_OPENAI_ENDPOINT"]
    openai.api_type = 'azure'
    openai.api_version = os.environ["OPENAI_API_VERSION"]
    deployment_name = 'gpt-4o-2024-05-13'

    response = openai.ChatCompletion.create(  # type: ignore
        engine=deployment_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token] if stop_token else None,
    )
    answer: str = response["choices"][0]["message"]["content"]
    end = time.time()
    print(f"Latency: {end - start}")
    return answer