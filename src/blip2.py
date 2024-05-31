from PIL import Image
import torch
from prompt import get_prompt, get_static_shots
from image_loader import load_images, load_knn_info
from utils import save_to_file
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm

knn_size=5
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_and_text(args):
    prompt = args['prompt']
    
    image_info = args['image_info']
    
    if args['few_shot_type'] == 'dynamic' and image_info['name'] in args['knn_info']:
        knn_info = args['knn_info'][image_info['name']]['knn']
    elif args['few_shot_type'] == 'static':
        knn_info = get_static_shots()
    else:
        knn_info = []

    images = []
    
    prompts = []
    for i, info in enumerate(knn_info):
        img_path = info['path']
        image = Image.open(img_path)
        images.append(image)
        prompts.append(f"The {i+1} photo was taken in {info['country']}")
    
    images.append(Image.open(image_info['path']))
    prompts.append(prompt)
    
    return images, prompt

def guess_country(args):
    model = args['model']
    processor = args['processor']
    
    images, prompt = get_image_and_text(args)
    
    inputs = processor(images, prompt, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=1024)
    return processor.decode(out[0], skip_special_tokens=True)


def excute_llm_task(images_info, params):    
    results = {}
    for i, info in enumerate(tqdm(images_info, 'Processing Blip2 Task')):
        name = info['name']
        params['image_info'] = info
        
        results[name] = guess_country(params)
        
        i = i + 1
        print(f"Processing image {i} / {len(images_info)}")
        print(f"Image name: {name}, {results[name]}")    
    return results


def run_llm_model(prompt, file_name, args, few_shot_type=None, randomized=False):
    images_info = load_images()
    params = {'prompt': prompt, 'few_shot_type': few_shot_type, 'randomized': randomized}
    params.update(args)
    
    results = excute_llm_task(images_info, params)
    save_to_file(results, file_name, 'json')

def get_model_and_processor(model_name):
    processor = Blip2Processor.from_pretrained(f"Salesforce/{model_name}")
    model = Blip2ForConditionalGeneration.from_pretrained(f"Salesforce/{model_name}", torch_dtype=torch.bfloat16).to(device)
    return model, processor


def run_model(model_name, posfix, pos_prompt=''):
    prompts = get_prompt()
    
    model, processor = get_model_and_processor(model_name)
    model, processor = get_model_and_processor(model_name)
    knn_info = load_knn_info(knn_size=knn_size)
    
    args = {'model': model, 'processor': processor, 'knn_info': knn_info}
    
    run_llm_model(f"{prompts['basic_prompt']}{pos_prompt}", f'basic/raw_basic_{posfix}', args)
    run_llm_model(f"{prompts['must_prompt']}{pos_prompt}", f'must/raw_basic_{posfix}', args)
    run_llm_model(prompts['blip2_tips_must_prompt'], f'tips/raw_tips_must_{posfix}', args)
    
    f_name=f'raw_must_{posfix}_{knn_size}'
    run_llm_model(f"{prompts['must_prompt']}{pos_prompt}", f'few_shots_dynamic/{f_name}', args, few_shot_type='dynamic', randomized=False)
    run_llm_model(f"{prompts['must_prompt']}{pos_prompt}", f'few_shots_dynamic/{f_name}_rd', args, few_shot_type='dynamic', randomized=True)
    run_llm_model(f"{prompts['must_prompt']}{pos_prompt}", f'few_shots_static/{f_name}', args, few_shot_type='static', randomized=False)
    run_llm_model(f"{prompts['must_prompt']}{pos_prompt}", f'few_shots_static/{f_name}_rd', args, few_shot_type='static', randomized=True)
    

def run_2_7b():
    # blip2-opt-2.7b
    posfix = 'blip2'
    model_name='blip2-opt-2.7b'
    run_model(model_name, posfix, ' Answer:')
    

def run_flan_t5():
    posfix = 'blip2-t5'
    model_name = 'blip2-flan-t5-xl'
    run_model(model_name, posfix)
    

if __name__ == "__main__":
    run_2_7b()
    run_flan_t5()    
