from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import torch
from prompt import get_prompt
from image_loader import load_images
from utils import save_to_file
from tqdm import tqdm

def guess_country(prompt, img_path, model, processor):
    image = Image.open(img_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
    generation_output = model.generate(**inputs, max_new_tokens=1024)
    generation_text = processor.batch_decode(generation_output[:, -1024:], skip_special_tokens=True)
    generation_text = generation_text[0].split("\x04\n\n")[-1]    
    
    return generation_text


def excute_fuyu_task(prompt, image_info):
    model_id = "adept/fuyu-8b"
    processor = FuyuProcessor.from_pretrained(model_id)
    model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)
    
    results = {}
    for i, info in tqdm(enumerate(image_info), 'Processing Fuyu Task'):
        name = info['name']
        img_path = info['path']
        results[name] = guess_country(prompt, img_path, model, processor)
        
        i = i + 1
        print(f"Processing image {i} / {len(image_info)}")
        print(f"Image name: {name}, {results[name]}")    
    return results


def run_fuyu(prompt, file_name):
    img_info = load_images()
    results = excute_fuyu_task(prompt, img_info)
    save_to_file(results, file_name, 'json')

if __name__ == "__main__":
    prompts = get_prompt()
    posfix = 'fuyu'
    run_fuyu(prompts['basic_prompt'], f'basic/raw_basic_{posfix}')
    run_fuyu(prompts['must_prompt'], f'must/raw_must_{posfix}')
    run_fuyu(prompts['tips_must_prompt'], f'tips/raw_tips_must_{posfix}')
