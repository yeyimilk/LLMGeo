import google.generativeai as genai
from image_loader import load_images
import PIL.Image
import time
from image_loader import load_images, load_knn_info
from utils import save_to_file
from prompt import get_prompt, get_static_shots
from tqdm import tqdm
import random
import os

knn_size = 5

def send_gen(content):
    attempt = 0
    
    while attempt < 3:
        try:
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content(content, stream=True)
            response.resolve()
            return response.text
        except BaseException as e:
            print(f"{attempt} {content} with error: {e}")
            
        time.sleep(2 ** (attempt + 1))
        attempt += 1
    
    return {'country': 'error'}
    
def get_static_dynamic_knn_info(prompt, img_path, img, few_shot, randomized=False):
    knn_info = load_knn_info(knn_size=knn_size)
    img_name = img_path.split('/')[-1]
        
    if few_shot == 'static' or img_name not in knn_info:
        knn_info = get_static_shots()
    else:
        knn_info = knn_info[img_name]['knn']
    
    if randomized:
        random.shuffle(knn_info)
        
    prompt = f"{prompt}\n===== Examples ====="
    content = []
    for i, info in enumerate(knn_info):
        country = info['country']
        sample = PIL.Image.open(info['path'])
        text = f"The following photo was taken in {country}"
        
        if i == 0:
            text = f"{prompt}\n{text}\n"
        else:
            text = f"{text}\n"
    
        content.append(text)
        content.append(sample)
    
    text = '==============\n Provide your answer for the following one'
    content.append(text)
    content.append(img)
    return content
    
def generate_content(prompt, info, few_shot, randomized=False):
    img_path = info['path']
    img = PIL.Image.open(img_path)
    
    if few_shot is None:
        return [prompt, img]
    elif few_shot == 'static' or few_shot == 'dynamic':
        return get_static_dynamic_knn_info(prompt, img_path, img, few_shot=few_shot, randomized=randomized)        
    else:
        raise ValueError(f"Invalid few_shot value: {few_shot}")
    
def excute_ai_task(prompt, image_info, few_shot, randomized):
    results = {}
    print(f"Processing {len(image_info)} images, few_shot: {few_shot}, randomized: {randomized}")
    for i, info in tqdm(enumerate(image_info)):
        
        name = info['name']
        content = generate_content(prompt, info, few_shot, randomized)
        results[name] = send_gen(content)
        
        i = i + 1
        print(f"Processing image {i} / {len(image_info)}")
        print(f"Image name: {name}, {results[name]}")
        if i % 10 == 0:
            time.sleep(10)
        
    return results


def run_gen(prompt, file_name, few_shot=None, randomized=False):
    GOOGLE_API_KEY = os.getenv('GOOGLE_AI_STUDIO_API_KEY') or 'YOUR_GEN_AI_KEY'
    genai.configure(api_key=GOOGLE_API_KEY)
    
    img_info = load_images()
    results = excute_ai_task(prompt, img_info, few_shot, randomized)
    save_to_file(results, file_name, 'json')

if __name__ == "__main__":
    prompts = get_prompt()
    run_gen(prompts['basic_prompt'], 'basic/raw_basic_gen')
    run_gen(prompts['must_prompt'], 'must/raw_must_gen')
    run_gen(prompts['tips_must_prompt'], 'tips/raw_tips_gen')
    
    run_gen(prompts['must_prompt'], f'few_shots_dynamic/raw_basic_gen_{knn_size}', 'dynamic')
    run_gen(prompts['must_prompt'], f'few_shots_static/raw_basic_gen_{knn_size}', 'static')
    run_gen(prompts['must_prompt'], f'few_shots_dynamic/raw_basic_gen_{knn_size}_rd', 'dynamic', True)
    run_gen(prompts['must_prompt'], f'few_shots_static/raw_basic_gen_{knn_size}_rd', 'static', True)
