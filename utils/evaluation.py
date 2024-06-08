import os
import sys
import json
sys.path.insert(0, '/home/getachew/Documents/10_Academy/Week_7/PrecisionRAG')
from openai import OpenAI
from utils.data_generator import get_completion,file_reader
# from data_generation._data_generation import file_reader
from dotenv import find_dotenv, load_dotenv
import numpy as np

env_file_path = find_dotenv(raise_error_if_not_found=True)
load_dotenv(env_file_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def evaluate(prompt: str, questions: str, context: str) -> str:
    """Return the classification of the hallucination.
    @parameter prompt: the prompt to be completed.
    @parameter user_message: the user message to be classified.
    @parameter context: the context of the user message.
    @returns classification: the classification of the hallucination.
    """
    num_test_output = str(10)
    API_RESPONSE = get_completion(
        [
            {
                "role": "system", 
                "content": prompt.replace("{Context}", context).replace("{Question}", questions)
            }
        ],
        model="gpt-3.5-turbo",
        logprobs=True,
        top_logprobs=1,
    )

    system_msg = str(API_RESPONSE.choices[0].message.content)

    for logprob in API_RESPONSE.choices[0].logprobs.content[0].top_logprobs:
        accuracy = np.round(np.exp(logprob.logprob)*100,2)
        output = f'\nhas_sufficient_context_for_answer: {system_msg}, \nlogprobs: {logprob.logprob}, \naccuracy: {np.round(np.exp(logprob.logprob)*100, 2)}%\n'
        print(output)
        if system_msg == 'true' and np.round(np.exp(logprob.logprob)*100, 2) >= 95.00:
            classification = 'true'
        elif system_msg == 'false' and np.round(np.exp(logprob.logprob)*100, 2) >= 95.00:
            classification = 'false'
        else:
            classification = 'false'
    result = {'classification': classification, 'accuracy': f'{accuracy}%'}
    return result


if __name__ == "__main__":
    context_message = file_reader("prompts/context.txt")
    prompt_message = file_reader("prompts/data_evaluation.txt")
    context = str(context_message)
    prompt = str(prompt_message)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Go one level up from the script's directory
    base_dir = os.path.dirname(script_dir)

    # Specify the file path relative to the base directory
    file_path = os.path.join(base_dir, "test_dataset/test_data.json")
    
    # Read the generated prompts file
    with open(file_path, 'r') as f:
        prompts = json.load(f)

    # Extract the questions from the prompts
    # questions = [prompts['prompt'] for prompt in prompts]
    questions = [item["user"] for item in prompts]

    print(prompts)
    print()
    
    # Evaluate each question
    for question in questions:
            print(evaluate(prompt, question, context))
    
       
# if __name__ == "__main__":
#     main()