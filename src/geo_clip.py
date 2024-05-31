
import torch
from geoclip import GeoCLIP
from image_loader import load_images
import numpy as np
from utils import save_to_file
import googlemaps
import json

key = "YOUR GOOGLE API KEY"
gmpas = googlemaps.Client(key=key)

# https://github.com/VicenteVivan/geo-clip?tab=readme-ov-file


def excute_geoclip_task(imgpath):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCLIP().to(device)
    top_pred_gps, top_pred_prob = model.predict(imgpath, top_k=10)
    top_pred_gps = top_pred_gps.tolist()
    top_pred_prob = top_pred_prob.tolist()
    ave_gps = np.mean(top_pred_gps, axis=0).tolist()
    
    return top_pred_gps, top_pred_prob, ave_gps

def run_geoclip():
    img_info = load_images()
        
    results = {}
    for info in img_info:
        top_pred_gps, top_pred_prob, ave_gps = excute_geoclip_task(info['path'])
        results[info['name']] = {
            'top_pred_gps': top_pred_gps,
            'top_pred_prob': top_pred_prob,
            'ave_gps': ave_gps
        }
    save_to_file(results, 'geoclip/raw_geoclip', 'json')


def reverse_geocode(key, lat, lon):
    try:
        results = gmpas.reverse_geocode((lat, lon))
        address_components = results[len(results)-1]['address_components']
        for item in address_components:
            if 'country' in item['types']:
                return item['long_name']
        print(f"Google can not find: {lat}, {lon}")

        with open(f'google_not_found_{key}.json', 'w') as f:
            json.dump(results, f, indent=4)

        return "unknown"
    except:
        with open(f'google_error_{key}.json', 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Google Error: {lat}, {lon}")
        return "unknown"


def transfer_geo_to_country(data_path):
    geoclip = json.load(open(data_path))

    results = {}

    for i, key in enumerate(geoclip.keys()):
        top_gps = geoclip[key]['top_pred_gps']
        gps = top_gps[0]
        result = geoclip[key]
        country = reverse_geocode(key, gps[0], gps[1])

        result['country'] = country

        results[key] = result
        print(f"Progress: {i + 1}/{len(geoclip.keys())}, => {country}")
    
    file_name = data_path.replace('geoclip', 'new_geoclip')
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)
        
        
if __name__ == "__main__":
    file_path = run_geoclip()
    transfer_geo_to_country(file_path)