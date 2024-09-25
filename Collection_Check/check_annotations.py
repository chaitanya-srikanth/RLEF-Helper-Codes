import pandas as pd
import json
import jsonify
from collections import Counter



df = pd.read_csv('dataSetCollection_testing-dataset-alpha_resources.csv')

all_list = []


for index in range(len(df)):
    try:
        structured_data = json.loads(df['imageAnnotations'][index])
    except:
        # dummy = 1
        print(df['name'][index])




    for data in structured_data:   
        
        try:
            all_list.append(data['selectedOptions'][1]['value'])
        except:
            print(df['name'][index])
            continue 

# print(all_list)

print(len(set(df['name'])))
print(len(df))
# 
print(Counter(all_list))
    
   
   