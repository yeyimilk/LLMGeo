import torch
from transformers import AutoTokenizer
import torch, auto_gptq
from auto_gptq.modeling._base import BaseGPTQForCausalLM
from utils import save_to_file
from prompt import get_prompt, get_static_shots
from image_loader import load_images, load_knn_info
import random
from PIL import Image
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
import argparse

knn_size=5
class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output', 
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]

def load_fine_tune_model(m_type):
    torch.set_grad_enabled(False)
    auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
    
    model_path = f'YOUR_FINETUNE_PATH TO: InternLM-XComposer/finetune/output/{m_type}'
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer
    
def load_model():
    torch.set_grad_enabled(False)
    auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
    # init model and tokenizer
    model_name = 'internlm/internlm-xcomposer2-vl-7b-4bit'
    model = InternLMXComposer2QForCausalLM.from_quantized(model_name, trust_remote_code=True, device="cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

def get_image_and_text(model, image_info, prompt, knn_info, few_shot_type=None, randomized=False):
    
    quess_img_name = image_info['name']
    
    if few_shot_type == 'dynamic' and quess_img_name in knn_info:
        knn_info = knn_info[quess_img_name]['knn']
    elif few_shot_type == 'static':
        knn_info = get_static_shots()
    else:
        knn_info = []
    
    if randomized:
        random.shuffle(knn_info)
    
    images = []
    query = ''
    
    a_text = ''
    for i, info in enumerate(knn_info):
        image = Image.open(info['path']).convert("RGB")
        image = model.vis_processor(image)
        images.append(image)
        query += '<ImageHere> '
        a_text += f"The {i+1} photo was taken in {info['country']}.\n"
    
    image = Image.open(image_info['path']).convert("RGB")
    image = model.vis_processor(image)
    images.append(image)
    image = torch.stack(images)
    
    query += '<ImageHere>'
    query = f"{query}{prompt}"
    
    if a_text != '':
        query = f"{query}\n===== Examples =====\n{a_text}\n==============\nProvide your answer for the last photo"
    
    return image, query

def excute_task(prompt, file_name, args, few_shot_type=None, randomized=False):    
    results = {}
    
    image_info = args['image_info']
    tokenizer = args['tokenizer']
    model = args['model']
    
    knn_info = load_knn_info(knn_size=knn_size)
    
    for info in tqdm(image_info, 'Processing InternLM2 Task'):
        name = info['name']
        
        image, text = get_image_and_text(model, info, prompt, knn_info, few_shot_type, randomized)
        
        with torch.cuda.amp.autocast(): 
            response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False) 
        
        results[name] = response
    
    save_to_file(results, file_name, 'json')
    return results

def main(m_type):
    if m_type == 'origin':
        model, tokenizer = load_model()
        prompt = get_prompt()
        image_info = load_images()
        posfix = 'internLM-7b'
    
    else:
        model, tokenizer = load_fine_tune_model(m_type)
        prompt = get_prompt()
        image_info = load_images()
        posfix = m_type
    
    args = {
        'image_info': image_info,
        'model': model,
        'tokenizer': tokenizer,
    }

    excute_task(prompt['basic_prompt'], f'basic/raw_basic_{posfix}', args)
    excute_task(prompt['must_prompt'], f'must/raw_must_{posfix}', args)
    excute_task(prompt['tips_must_prompt'], f'tips/raw_tips_must_{posfix}', args)
    
    if m_type == 'origin':
        f_name = f'raw_must_{posfix}_{knn_size}'
        excute_task(prompt['must_prompt'], f'few_shots_static/{f_name}', args, few_shot_type='static', randomized=False)
        excute_task(prompt['must_prompt'], f'few_shots_static/{f_name}_rd', args, few_shot_type='static', randomized=True)
        excute_task(prompt['must_prompt'], f'few_shots_dynamic/{f_name}', args, few_shot_type='dynamic', randomized=False)
        excute_task(prompt['must_prompt'], f'few_shots_dynamic/{f_name}_rd', args, few_shot_type='dynamic', randomized=True)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='InternLM-XComposer')
    parser.add_argument('--model', type=str, default='1', help='model type, 1: origin, 2: finetune 3K, 3: finetune 6K')
    args = parser.parse_args()
    models = {
        '1': 'origin',
        '2': 'internLM-7b-3K',
        '3': 'internLM-7b-6K',
    }
    main(models[args.model])
