import os
import pathlib
from datetime import datetime
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def make_sure_folder_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def make_path(name, time_t, no_result):
    if time_t:
        name = name_with_datetime(name)
    abs_path = pathlib.Path(__file__).parent.resolve()
    
    sub_path = os.path.dirname(name)
    
    file_dir = ""
    if no_result:
        file_dir = f'{abs_path}/{sub_path}'
    else:
        file_dir = f'{abs_path}/results/{sub_path}'
    
    make_sure_folder_exists(file_dir)
    return abs_path, name, file_dir

def save_to_file(content, name, type='txt', time_t=True, no_result=False):
    abs_path, name, file_dir = make_path(name, time_t, no_result)
    
    if type == 'json':
        content = json.dumps(content, cls=NpEncoder)
    
    file_path = ''
    if no_result:
        file_path = f'{abs_path}/{name}.{type}'
    else:
        file_path =  f'{abs_path}/results/{name}.{type}'
        
    with open(file_path, 'w') as f:
        f.write(content)
    
    return file_path


def get_all_json_files_from_folder(directory):
    json_files = []
    
    # Walk through all directories and files in 'directory'
    for dirpath, dirnames, filenames in os.walk(directory):
        # Filter for json files and join the path and file name
        json_files.extend([os.path.join(dirpath, f) for f in filenames if f.endswith('.json')])
    
    json_files.sort()
    return json_files
        