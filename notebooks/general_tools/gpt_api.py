import base64
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import AzureOpenAI


def get_azure_client(dotenv_path: str = ".env") -> AzureOpenAI:
    """
    Create Azure OpenAI client using credentials from .env
    """
    load_dotenv(dotenv_path)

    endpoint = os.getenv("azure_endpoint")
    key = os.getenv("azure_key")
    api_version = os.getenv("azure_api_version", "2025-03-01-preview")

    if not endpoint:
        raise ValueError("azure_endpoint not found in .env")
    if not key:
        raise ValueError("azure_key not found in .env")

    azure_client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=api_version,
    )

    print("Azure client initiated!")
    return azure_client

def _image_path_to_block(image_path: str) -> Dict[str, str]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "type": "input_image",
        "image_url": f"data:image/jpg;base64,{b64}",
    }

def azure_chat_completion(
    deployment: str,
    messages: List[Dict[str, Any]],
    *,
    dotenv_path: str = ".env",
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    **kwargs: Any,
) -> str:
    """
    Azure OpenAI Responses API call

    Each message may optionally include:
      - image_path: str
      - image_paths: list[str]

    If present, image(s) are appended to that message's content blocks.
    """

    client = get_azure_client(dotenv_path)

    input_payload: List[Dict[str, Any]] = []

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")

        # normalize content into responses blocks
        if isinstance(content, str):
            blocks: List[Dict[str, Any]] = [{"type": "input_text", "text": content}]
        elif isinstance(content, list):
            blocks = content
        else:
            raise TypeError("message content must be str or list of blocks")

        # collect images for this message
        img_paths: List[str] = []
        if m.get("image_path"):
            img_paths.append(m["image_path"])
        if m.get("image_paths"):
            img_paths.extend(m["image_paths"])

        # append image blocks
        for p in img_paths:
            blocks.append(_image_path_to_block(p))

        input_payload.append({"role": role, "content": blocks})

    resp = client.responses.create(
        model=deployment,
        input=input_payload,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **kwargs,
    )

    return resp.output_text