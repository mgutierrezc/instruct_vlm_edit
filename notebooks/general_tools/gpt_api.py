import os
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from openai import AzureOpenAI


def get_azure_client(dotenv_path: str = ".env") -> AzureOpenAI:
    """
    Create Azure OpenAI client using credentials from .env
    """
    load_dotenv(dotenv_path)

    endpoint = os.getenv("azure_endpoint")
    key = os.getenv("azure_key")
    api_version = os.getenv("azure_api_version", "2024-02-15-preview")

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

def azure_chat_completion(
    deployment: str,
    messages: List[Dict[str, str]],
    *,
    dotenv_path: str = ".env",
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    **kwargs: Any,
) -> str:
    """
    Azure OpenAI chat call using deployment name.
    """

    client = get_azure_client(dotenv_path)

    # response = client.responses.create(
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )

    return response.choices