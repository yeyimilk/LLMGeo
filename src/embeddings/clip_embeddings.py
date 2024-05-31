from transformers import AutoProcessor, CLIPModel
from PIL import Image
import torch 
from image_loader import load_images
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def generate_embeddings(imgs, model, processor, f_path):
    embeddings = list()
    with torch.no_grad() : 
        for img in tqdm(imgs) : 
            name = img['name']
            path = img['path']
            image = Image.open(path)
                
            input_preprocessed = processor(images=image, return_tensors="pt").to(device)
            output = model.get_image_features(**input_preprocessed)
            embeddings.append({
                "name" : name,
                "path" : path,
                "embedding" : output[0].detach().cpu().numpy(),
            })

    with open(f_path, "wb") as f:
        np.save(f, embeddings)


def run_clip():
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    
    imgs = load_images("train")
    generate_embeddings(imgs, model, processor, '../dataset/embeddings/clip/train.npy')

    imgs = load_images("test")
    generate_embeddings(imgs, model, processor, '../dataset/embeddings/clip/test.npy')

if __name__ == "__main__":
    run_clip()
