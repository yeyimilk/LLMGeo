
import json
from utils import get_all_json_files_from_folder
from image_loader import load_images

def re_format_country(country):
    country = country.lower()
    if country == "united states of america" or country == "usa" or country == "united states of america (u.s.)" or country == "us":
        country = "united states"
    if country == "england" or country == 'uk':
        country = "united kingdom"
    return country

def check_result(predicted_path):
    print(f"--------{predicted_path}----------\n")
    
    labels = load_images()
    basic = json.load(open(predicted_path))

    corrected = 0
    unknown = 0
    wrong = 0

    values = []

    for info in labels:
        key = info['name']
        country_data = basic[key]
        if type (country_data) != dict:
            try:
                country_data = json.loads(country_data)
            except:
                try:
                    country_data = country_data.split(':')[1]
                    country_data = country_data.split('}')[0].strip()
                    country_data = {"country": country_data}
                except:
                    country_data = {"country": 'unknown'}
        
        predicted_country = re_format_country(country_data['country'])
        gt_country = info['country'].lower()
        
        values.append(predicted_country)
        
        if predicted_country == 'unknown':
            unknown += 1
        elif predicted_country == gt_country:
            corrected += 1
        else:
            wrong += 1
    
    total = corrected + unknown + wrong
    
    print(f'\ntotal: {total} corrected: {corrected}, unknown: {unknown}, wrong: {wrong}')
    print(f'corrected rate: {corrected / total}')
    print(f'unknown rate: {unknown / total}')
    print(f'wrong rate: {wrong / total}')
    total_neg_un = 0 if total == unknown else corrected / (total - unknown)
    print(f'corrected rate without unknown: {total_neg_un}')
    
    print("=================================\n")
    return corrected / total, unknown / total, wrong / total, total_neg_un, values
    
def check_results(directory):
    json_files = get_all_json_files_from_folder(directory)
    results = []
    
    values = []
    for json_file in json_files:
        acc, unknown, wrong, acc_n_unknown, country_values = check_result(json_file)
        results.append([acc, unknown, wrong, acc_n_unknown])
        values += country_values

if __name__ == '__main__':
    check_results('./cleaned_results/llm_cleaned')
