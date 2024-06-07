import os
import json
import sys
from openai import OpenAI
from math import exp
import numpy as np
import logging
from PyPDF2 import PdfReader
from main import get_env_manager
env_manager = get_env_manager()
client = OpenAI(api_key=env_manager['openai_keys']['OPENAI_API_KEY'])


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def get_completion(
    messages: list[dict[str, str]],
    model: str = env_manager['vectordb_keys']['VECTORDB_MODEL'],
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> str:
    """Return the completion of the prompt.
    @parameter messages: list of dictionaries with keys 'role' and 'content'.
    @parameter model: the model to use for completion. Defaults to 'davinci'.
    @parameter max_tokens: max tokens to use for each prompt completion.
    @parameter temperature: the higher the temperature, the crazier the text
    @parameter stop: token at which text generation is stopped
    @parameter seed: random seed for text generation
    @parameter tools: list of tools to use for post-processing the output.
    @parameter logprobs: whether to return log probabilities of the output tokens or not.
    @returns completion: the completion of the prompt.
    """

    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion


def file_reader(path):
    """
    Read the contents of a file.
    
    Args:
        path (str): The path to the file.
    
    Returns:
        str: The content of the file.
    """
    try:
        if path.endswith('.txt'):
            with open(path, 'r') as f:
                content = f.read()
            logger.info(f"File reading successful for {path}")
        elif path.endswith('.pdf'):
            content = ""
            with open(path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    content += page.extract_text()
            logger.info(f"PDF reading successful for {path}")
        else:
            raise ValueError("Unsupported file format")
        return content
    except Exception as e:
        logger.error(f"Error reading the file: {e}")
        raise


def generate_test_data(prompt: str, context: str, num_test_output: str) -> str:
    """Return the classification of the hallucination.
    @parameter prompt: the prompt to be completed.
    @parameter user_message: the user message to be classified.
    @parameter context: the context of the user message.
    @returns classification: the classification of the hallucination.
    """
    API_RESPONSE = get_completion(
        [
            {
                "role": "user", 
                "content": prompt.replace("{context}", context).replace("{num_test_output}", num_test_output)
            }
        ],
        model=env_manager['vectordb_keys']['VECTORDB_MODEL'],
        logprobs=True,
        top_logprobs=1,
    )

    system_msg = API_RESPONSE.choices[0].message.content
    return system_msg


def split_text_into_chunks(text, chunk_size=200):
    """
    Split text into chunks of a specified size.
    
    Args:
        text (str): The text to split.
        chunk_size (int): The size of each chunk.
    
    Returns:
        list: A list of text chunks.
    """
    try:
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        logger.info(f"Successfully chunked text into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking the text: {e}")
        raise


def main(num_test_output: str):
    context_message = file_reader("prompts/context.txt")
    prompt_message = file_reader("prompts/data_generation.txt")
    context = str(context_message)
    prompt = str(prompt_message)
    test_data = generate_test_data(prompt, context, num_test_output)
    def save_json(test_data) -> None:
        # Specify the file path
        file_path = "test_dataset/test_data.json"
        json_object = json.loads(test_data)
        with open(file_path, 'w') as json_file:
            json.dump(json_object, json_file, indent=4)
            
        print(f"JSON data has been saved to {file_path}")

    save_json(test_data)

    print("===========")
    print("Test Data")
    print("===========")
    print(test_data)

if __name__ == "__main__":
    main("5") # n number of test data to generate