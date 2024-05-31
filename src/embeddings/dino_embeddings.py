from transformers import AutoImageProcessor, AutoModel
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
            output = model(**input_preprocessed)
            image_features = output.last_hidden_state
            image_features = image_features.mean(dim=1)
            embeddings.append({
                "name" : name,
                "path" : path,
                "embedding" : image_features[0].detach().cpu().numpy(),
            })

    with open(f_path, "wb") as f:
        np.save(f, embeddings)

def run_dino():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    model.eval()
    
    imgs = load_images("train")
    generate_embeddings(imgs, model, processor, '../dataset/embeddings/dino/train.npy')

    imgs = load_images("test")
    generate_embeddings(imgs, model, processor, '../dataset/embeddings/dino/test.npy')

if __name__ == '__main__':
    run_dino()
