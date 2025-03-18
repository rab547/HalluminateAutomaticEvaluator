import os
from typing import Dict, List

from dotenv import load_dotenv

import litellm

load_dotenv()  # Load environment variables from .env file

MODEL_LIST = [
    {
        "model_name": "o1-azure",
        "litellm_params": {
            "model": "azure/o1",
            "api_key": os.getenv("AZURE_OPENAI_API_KEY_O1_AND_O3"),
            "api_base": os.getenv("O1_URL"),
            "api_version": "2024-12-01-preview",
        },
    },
    {
        "model_name": "o3-mini-azure",
        "litellm_params": {
            "model": "azure/o3-mini",
            "api_key": os.getenv("AZURE_OPENAI_API_KEY_O1_AND_O3"),
            "api_base": os.getenv("O3_MINI_URL"),
            "api_version": "2024-12-01-preview",
        },
    }
]

ROUTER = litellm.Router(
    model_list=MODEL_LIST,
    num_retries=2,
    fallbacks=[
        #flashtink to llama
    ],
)

def generate_response(
    model: str,
    messages: List[Dict[str, str]],
    **kwargs,
) -> str:
    """Generate a response using the LLM.

    Args:
        model: The model to use for generation
        messages: List of message dictionaries with 'role' and 'content'
        **kwargs: Additional arguments to pass to the completion API
    """
    response = ROUTER.completion(
        model=model,
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content
