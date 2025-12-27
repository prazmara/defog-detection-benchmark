"""Azure OpenAI client setup and configuration."""

import os
from typing import Tuple

import dotenv
from openai import AzureOpenAI

from tools.gpt_endpoint.gpt_tool_azure import AzureVLMScorer
from pipeline.config import AZURE_API_VERSION, AZURE_ENDPOINT, DEPLOYMENT_NAME


def setup_azure_clients() -> Tuple[AzureOpenAI, AzureVLMScorer]:
    """Initialize Azure OpenAI clients and VLM scorer."""
    dotenv.load_dotenv()
    
    subscription_key = os.getenv("GPT5_CHAT")
    if not subscription_key:
        raise ValueError("GPT5_CHAT environment variable not set")
    
    client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=subscription_key,
    )
    
    scorer = AzureVLMScorer(
        client,
        deployment=DEPLOYMENT_NAME,
        temperature=0.0
    )
    
    return client, scorer


def setup_output_directory(output_csv: str) -> None:
    """Create output directory if it doesn't exist."""
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
