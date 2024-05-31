
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import save_to_file
from code_to_country import code_to_country
import json
from embeddings import run_clip, run_dino

def load_embedding(path):
    f_path = '../dataset/embeddings/' + path + '.npy'
    with open(f_path, "rb") as f:
        embedding = np.load(f, allow_pickle=True)
    return embedding


def load_embeddings(type):
    return load_embedding(type + '/train'), load_embedding(type + '/test')

def knn_for_embeddings(train, test, k=2):
    train_embeddings = np.array([x['embedding'] for x in train])
    
    results = {}
    for v in test:
        name = v['name']
        embedding = v['embedding']
        
        knn_model = NearestNeighbors(n_neighbors=k)
        knn_model.fit(train_embeddings)
        _, indices = knn_model.kneighbors([embedding])
    
        nn_from_train = []
        for i in indices[0]:
            code = train[i]['path'].split('/')[4]
            country = code_to_country(code)
            nn_from_train.append({
                'name': train[i]['name'],
                'path': train[i]['path'],
                'code': code,
                'country': country,
            }) 
        
        results[name] = {
            'path': v['path'],
            'knn': nn_from_train
        }
    
    return results


def check_acc(name, results):
    correct_1 = 0
    correct_2 = 0
    for item in results:
        item = results[item]
        knns = [knn['code'] for knn in item['knn'][0: 5]]
        y_true = item['path'].split('/')[4]
        
        if y_true == knns[0]:
            correct_1 += 1

        if y_true in knns:
            correct_2 += 1
            
    print(f"Results for {name}")
    print(f"Accuracy@1: {correct_1 / len(results)}")
    print(f"Accuracy@5: {correct_2 / len(results)}\n")
    
def run_knn():
    n = 5
    train, test = load_embeddings('dino')
    results = knn_for_embeddings(train, test, n)
    save_to_file(results, f'../dataset/embeddings/dino/knn_{n}', 'json', time_t=False, no_result=True)
    check_acc('dino', results)
    
    train, test = load_embeddings('clip')
    results = knn_for_embeddings(train, test, n)
    save_to_file(results, f'../dataset/embeddings/clip/knn_{n}', 'json', time_t=False, no_result=True)
    check_acc('clip', results)

def check_results():
    dino = json.load(open('../dataset/embeddings/dino/knn_5.json'))
    check_acc('dino', dino)
    
    clip = json.load(open('../dataset/embeddings/clip/knn_5.json'))
    check_acc('clip', clip)

if __name__ == '__main__':
    run_clip()
    run_dino()
    run_knn()
    check_results()