
import json

def load_countries_dict():
    json_file = open('../dataset/countries.json')
    countries = json.load(json_file)
    countries_dict = {}
    for country in countries:
        countries_dict[country['code']] = country['name']
    
    return countries_dict
    

def code_to_country(code):
    country_dict = load_countries_dict()
    code = code.upper()
    
    if code in country_dict:
        return country_dict[code]
    
    return 'unkown'