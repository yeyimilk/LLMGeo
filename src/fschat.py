
from utils import save_to_file, get_all_json_files_from_folder
import torch
import json
from fastchat.model import load_model, get_conversation_template
from tqdm import tqdm
import os

def generate_result(msg, model_path, model, tokenizer):
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        temperature=0,
        repetition_penalty=1.0,
        max_new_tokens=1024,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    
    return outputs

def get_dest_path(s_path):
    return s_path.replace('raw_', '').replace('results', 'cleaned_results/llm_cleaned').replace('.json', '').replace('./', '')


def get_prompt(content):
    txt = "A photo is provided for a LLM agent to identify where the country is. But not all the answers provided in a standard format. Please extract the correct country from the content and formated it into a standard name. You should not provide a new country from the content. Your answer should be in json format, {country: country_name}. If the country is not provided, please return {country: unknown}"
    return f"{txt}\n---Content---\n{content}"    

@torch.inference_mode()
def translate_country(json_files=[], dests=[]):
    model_path = "lmsys/vicuna-13b-v1.5"
    model, tokenizer = load_model(model_path)
    
    if json_files == []:
        json_files = get_all_json_files_from_folder("./results")
    
    for i, j_file in tqdm(enumerate(json_files), desc="Processing JSON files"):
        if len(dests) and len(dests) > i and dests[i]:
            dest_path = dests[i]
        else:
            dest_path = get_dest_path(j_file)
            
        if os.path.isfile(f"{dest_path}.json"):
            continue
        
        print(f"Start handling {j_file}")                  

        data = json.load(open(j_file))
        results = {}
        for key in tqdm(data.keys(), desc=f"Processing keys in JSON file", leave=False):
            msg = data[key]
            
            if type(msg) is str and msg.strip() == "":
                results[key] = {"country": "unknown"}
            else:
                prompt = get_prompt(msg)
                results[key] = generate_result(prompt, model_path, model, tokenizer)

        
        save_to_file(results, dest_path, 'json', time_t=False, no_result=True)

if __name__ == "__main__":
    translate_country()
