import os
import json
import sys

current_directory = os.getcwd()
print(current_directory)
sys.path.insert(0, '/home/getachew/Documents/10_Academy/Week_7/PrecisionRAG')
# from data_retriver import retrieve_context, retriever

# project_root = os.path.dirname(os.path.abspath(__file__))
# os.chdir(project_root)

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
# from data_generation.retrive import retrieve_context

env_file_path = find_dotenv(raise_error_if_not_found=True)
load_dotenv(env_file_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)



def get_completion(messages, model="gpt-3.5-turbo", max_tokens=500, temperature=0, stop=None, seed=123, tools=None, logprobs=None, top_logprobs=None):
    params = {
        "messages": messages,
        "model": model,
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


    fname = os.path.join( path)
    with open(fname, 'r') as f:
        return f.read()

def generate_test_data(prompt, context, num_test_output):
    API_RESPONSE = get_completion([
        {"role": "user", "content": prompt.replace("{context}", context).replace("{num_test_output}", num_test_output)}
    ], logprobs=True, top_logprobs=1)

    return API_RESPONSE.choices[0].message.content

def save_json(test_data):
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory=os.path.dirname(current_script_directory)
    print(parent_directory)
    file_path = os.path.join(parent_directory,"test_dataset/test_data.json")
    json_object = json.loads(test_data)
    with open(file_path, 'w') as json_file:
        json.dump(json_object, json_file, indent=4)

    print(f"JSON data has been saved to {file_path}")

def main(num_test_output):
    # context_message=context=retrieve_context(inputText)
  

# Get the parent directory of the current script directory
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory=os.path.dirname(current_script_directory)
    # print(parent_directory)
    folder_path = "prompts/"
    file_path = os.path.join(folder_path, 'context.txt')

    print(file_path)
    context= file_reader(file_path)
    # context= file_reader(os.path.join(parent_directory,"prompts/context.txt"))
    prompt = file_reader(os.path.join(parent_directory,"prompts/data_generation.txt"))
    test_data = generate_test_data(prompt, context, num_test_output)
    save_json(test_data)

    print("===========")
    print("Test Data")
    print("===========")
    print(test_data)

if __name__ == "__main__":
    # user_input = str(input("inputText: "))
    # context=retrieve_context()
    main("3")  # n number of test data to generateor