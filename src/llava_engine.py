
from llava.eval.run_llava import get_model_and_processor, run_for_outputs
from types import SimpleNamespace

import random
from image_loader import load_images, load_knn_info
from prompt import get_prompt, get_static_shots
from utils import save_to_file
from tqdm import tqdm
import argparse

knn_size=5

def guess_country(prompt, img_path, cfig):
    args = SimpleNamespace(query=prompt, 
                           image_file=img_path, 
                           sep=",", temperature=0, 
                           top_p=None, num_beams=1, max_new_tokens=512)
    return run_for_outputs(cfig, args)
    
def generate_prompt_and_image_path(prompt, img_info, knn_info, few_shot, randomized=False):
    if few_shot is None:
        return prompt, img_info['path']
    
    name = img_info['name']
    img_path = img_info['path']
    
    imgs_path = ''
    prompt = f"{prompt}\n===== Examples ====="
    if few_shot == 'dynamic' and name in knn_info:
        knn_info = knn_info[name]['knn']
    else:
        knn_info = get_static_shots()
    
    if randomized:
        random.shuffle(knn_info)
        
    for i, info in enumerate(knn_info):
        prompt = f"{prompt}\nThe {i+1} photo was taken in {info['country']}"
        if imgs_path == '':
            imgs_path = info['path']
        else:
            imgs_path = f"{imgs_path},{info['path']}"
    
    prompt = f"{prompt}\n==============\n Provide your answer for the last photo"
    imgs_path = f"{imgs_path},{img_path}"

    return prompt, imgs_path
        
    
def excute_llava_task(prompt, image_info, model_params, few_shot, randomized):
    results = {}
    model_path, base_model = model_params
    model, image_processor, tokenizer, p_conv_mode = get_model_and_processor(model_path, base_model, None)
    
    cfig = {"model": model, "image_processor": image_processor, "tokenizer": tokenizer, "p_conv_mode": p_conv_mode}
    knn_info = load_knn_info(knn_size=knn_size)
    
    for info in tqdm(image_info, 'Processing LLaVA Task'):
        name = info['name']
        
        new_prompt, imgs_path = generate_prompt_and_image_path(prompt, info, knn_info, few_shot, randomized)
        results[name] = guess_country(new_prompt, imgs_path, cfig)
    
    return results

def run_llava(prompt, file_name, model_params, few_shot=None, randomized=False):
    img_info = load_images()
    results = excute_llava_task(prompt, img_info, model_params, few_shot, randomized)
    save_to_file(results, file_name, 'json')


def run_official_llava(model_type):
    model_path = f"liuhaotian/llava-v1.5-{model_type}"
    model_params = (model_path, None)
    
    prompts = get_prompt()
    run_llava(prompts['basic_prompt'], f'basic/raw_basic_llava_{model_type}', model_params)
    run_llava(prompts['must_prompt'], f'must/raw_must_llava_{model_type}', model_params)
    run_llava(prompts['tips_must_prompt'], f'tips/raw_tips_must_llava_{model_type}', model_params)
    
    run_llava(prompts['must_prompt'], f'few_shots_dynamic/raw_must_llava_{model_type}_{knn_size}', model_params=model_params, few_shot='dynamic', randomized=False)
    run_llava(prompts['must_prompt'], f'few_shots_static/raw_must_llava_{model_type}_{knn_size}', model_params=model_params, few_shot='static', randomized=False)
    run_llava(prompts['must_prompt'], f'few_shots_dynamic/raw_must_llava_{model_type}_{knn_size}_rd', model_params=model_params, few_shot='dynamic', randomized=True)
    run_llava(prompts['must_prompt'], f'few_shots_static/raw_must_llava_{model_type}_{knn_size}_rd', model_params=model_params, few_shot='static', randomized=True)
    

def run_finetune_llava(model_type):
    pre_path = 'YOUR_PATH_TO_FINE_TUNE:~/LLaVA/checkpoints'
    prompts = get_prompt()
    
    models = {
        '1': 'llava-v1.5-7b-3k',
        '2': 'llava-v1.5-7b-6k',
        '3': 'llava-v1.5-13b-3k',
        '4': 'llava-v1.5-13b-6k',
    }
    
    m_nmae = models[model_type]
    model_path = f"{pre_path}/{m_nmae}"
    
    base_model = '7b' if '7b' in m_nmae else '13b'
    base_model = f"liuhaotian/llava-v1.5-{model_type}"

    model_params = (model_path, base_model)
    
    run_llava(prompts['basic_prompt'], f'basic/raw_basic_{m_nmae}', model_params, few_shot=None, randomized=False)
    run_llava(prompts['must_prompt'], f'must/raw_must_{m_nmae}', model_params, few_shot=None, randomized=False)
    run_llava(prompts['tips_must_prompt'], f'tips/raw_tips_must_{m_nmae}', model_params, few_shot=None, randomized=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLaVA 1.5')
    parser.add_argument('--model', type=str, default='7b', help='7b and 13b is the original model, otherwise is the finetuned model')
    args = parser.parse_args()
    
    model = args.model
    if model == '7b' or '13b':
        run_official_llava(model)
    else:
        run_finetune_llava(model)
    
    

    
    
    
    
    

            
