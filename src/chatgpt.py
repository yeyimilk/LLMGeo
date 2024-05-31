
import openai
import time
import random
import base64
import os
from image_loader import load_images
from utils import save_to_file
from prompt import get_prompt
from tqdm import tqdm

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 3,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = (None, None),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                delay *= exponential_base * (1 + jitter * random.random())

                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper

@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def send_to_gpt(messages, model='gpt-4-vision-preview'):
    max_retry_attempts = 3

    for i in range(max_retry_attempts):
        try:
            response = completions_with_backoff(
                    model=model,
                    messages=messages,
                    # response_format = {"type": 'json_object'}, not supported yet
                    max_tokens=4096,
            )
            content = response['choices'][0]['message']['content']
            # content = json.loads(content)
            return content
        except Exception as e:
            print(f"An unexpected error occurred on {i+1} time: {e}")
            time.sleep(10)
    
    return {'country': 'error'}

def get_prompt_message(prompt, img_path):
    with open(img_path, 'rb') as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
        
    prompt_message = [{
        "role": "user",
        "content": [
            {
                'type': 'text',
                'text': prompt
            },
            {
                'type': 'image_url',
                'image_url': {
                    "url": f"data:image/jpeg;base64,{encoded_img}"
                }
            }
        ]
    }]
    
    return prompt_message
    
def excute_ai_task(prompt, image_info):
    results = {}

    
    for i, info in tqdm(enumerate(image_info), 'Processing ChatGPT Task'):
        
        name = info['name']
        img_path = info['path']
        
        prompt_message = get_prompt_message(prompt, img_path)
        results[name] = send_to_gpt(prompt_message)
        
        i = i + 1        
        if i % 10 == 0:
            time.sleep(10)
    
    return results


def run_chatgpt(prompt, file_name):
    openai.api_key = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
    
    img_info = load_images()
    results = excute_ai_task(prompt, img_info)
    save_to_file(results, file_name, 'json')
    

if __name__ == '__main__':
    prompts = get_prompt()
    posfix = 'chatgpt4'
    run_chatgpt(prompts['basic_prompt'], f'basic/raw_basic_{posfix}')
    run_chatgpt(prompts['must_prompt'], f'must/raw_must_{posfix}')
    run_chatgpt(prompts['tips_must_prompt'], f'tips/raw_tips_must_{posfix}')