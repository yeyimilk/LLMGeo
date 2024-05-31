import os
import glob
import math
import json
from utils import save_to_file

from code_to_country import load_countries_dict

def load_images(sub_path='test', with_additional=False):
    image_directory = f"../dataset/{sub_path}"
    image_paths = glob.glob(os.path.join(image_directory, "**", '*.jpg'), recursive=True)

    countries_dict = load_countries_dict()
    
    image_info = []
    
    for path in image_paths:
        if 'comprehensive' in path and with_additional is False:
            continue
        
        code = path.split("/")[-2].upper()
        name = os.path.basename(path)
        image_info.append({
            "name": name,
            "path": path,
            "code": code,
            "country": countries_dict.get(code, "unknown"),
            "longitude": float(name.split("_")[0]),
            "latitude": float(name.split("_")[1]),
        })
    
    print(f"Found {len(image_info)} images")

    return image_info


def compare_locations(locations):
    close_pairs = []  # List to hold pairs of locations that meet the criteria

    # Iterate through each pair of locations
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):  # Start from i+1 to avoid comparing a location with itself and repeating comparisons
            lat_diff = abs(locations[i]['latitude'] - locations[j]['latitude'])
            long_diff = abs(locations[i]['longitude'] - locations[j]['longitude'])
            
            close_pairs.append({
                "name1": locations[i]['name'],
                "name2": locations[j]['name'],
                "distance": math.sqrt(lat_diff**2 + long_diff**2) * 111
            })

    return close_pairs


def load_knn_info(data_set='clip', knn_size=2):
    return json.load(open(f'../dataset/embeddings/{data_set}/knn_{knn_size}.json', 'r'))


def distances_statistic(images):
    closed_images = compare_locations(images)

    dis_10 = 0
    dis_100 = 0
    dis_500 = 0
    dis_1000 = 0
    dis_others = 0
    for pair in closed_images:
        if pair['distance'] < 10:
            dis_10 += 1
        elif pair['distance'] < 100:
            dis_100 += 1
        elif pair['distance'] < 500:
            dis_500 += 1
        elif pair['distance'] < 1000:
            dis_1000 += 1
        else:
            dis_others += 1
    
    print("Distance between image pairs:")
    print(f"Distance < 10 km: {dis_10}, {dis_10/len(closed_images) * 100:.4f}%")
    print(f"10 =< Distance < 100: {dis_100}, {dis_100/len(closed_images) * 100:.2f}%")
    print(f"100 =< Distance < 500: {dis_500}, {dis_500/len(closed_images) * 100:.2f}%")
    print(f"500 =< Distance < 1000: {dis_1000}, {dis_1000/len(closed_images) * 100:.2f}%")
    print(f"Distance >= 1000: {dis_others}, {dis_others/len(closed_images) * 100:.2f}%")


def make_intern_dataset(data, f_name):
    data_list = []
    for i, info in enumerate(data):
        img_path = info['path']
        # you may need to change the path
        # img_path = img_path.replace('..', 'ABS_PATH/projects/llm-geoguess')
        data_list.append({
            "id": f"{i}",
            "image": [
                img_path
            ],
            "conversations": [
                {
                    "from": "user",
                    "value": "<ImageHere> Which country the image was taken in"
                },
                {
                    "from": "assistant",
                    "value": f"{info['country']}"
                }
            ]
        })
    save_to_file(data_list, f"../dataset/finetunes/{f_name}", 'json', time_t=False, no_result=True)

def make_llava_dataset(data, f_name):
    data_list = []
    for i, info in enumerate(data):
        img_path = info['path']
        # you may need to change the path
        # img_path = img_path.replace('..', 'ABS_PATH/projects/llm-geoguess')
        data_list.append({"id": info['name'], 
                          "image": img_path, 
                          "conversations": [{"from": "human", "value": "<image>\nWhich country the image was taken in"}, 
                                            {"from": "gpt", "value": f"{info['country']}"}]})
    save_to_file(data_list, f"../dataset/finetunes/{f_name}", 'json', time_t=False, no_result=True)

if __name__ == '__main__':
    test_images = load_images('test')
    train_images = load_images('train', with_additional=False)
    comprehensive_images = load_images('train', with_additional=True)
    
    make_intern_dataset(train_images, 'intern_train_d')
    make_intern_dataset(comprehensive_images, 'intern_train_all_d')
    
    make_llava_dataset(train_images, 'llava_train_d')
    make_llava_dataset(train_images, 'llava_train_all_d')

    