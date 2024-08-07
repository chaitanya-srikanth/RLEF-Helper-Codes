import os
import requests
from datetime import date
import json
import random
import math 
import operator
from functools import reduce
import traceback
import os
# import matplotlib.pyplot as plt
# import cv2
import numpy as np
from glob import glob
# import threading
from threading import Thread
import pandas as pd

MAX_THREADS = 100
# 65dca777daed8358d2482b6a

# path to Weird format
def pointPath_To_WeirdAutoaiAnnotationFormat(bboxes, labels):
    # creating dict for annotation
        li = {}

        # print(bboxes)
        # print(labels)

        for mask, label in zip(bboxes, labels):
            annotString = "["

            for x, y in mask:
                annotString += f"[{str(int(x))}, {str(int(y))}], "
            
            annotString += f"[{str(int(mask[-1][0]))}, {str(int(mask[-1][1]))}]]"

            li[annotString] = label

        # print(li)
        
        return li


# Json creater function for annotation of RLEF.ai
def json_creater(inputs, closed):

    # print(inputs)
    data = []
    count = 1
    highContrastingColors = ['rgba(0,255,81,1)', 'rgba(255,219,0,1)', 'rgba(255,0,0,1)', 'rgba(0,4,255,1)',
                             'rgba(227,0,255,1)']
    for index, input in enumerate(inputs):
        color = random.sample(highContrastingColors, 1)[0]
        json_id = count
        sub_json_data = {}
        sub_json_data['id'] = json_id
        sub_json_data['name'] = json_id
        sub_json_data['color'] = color
        sub_json_data['isClosed'] = closed
        sub_json_data['selectedOptions'] = [{"id": "0", "value": "root"},
                                            {"id": str(random.randint(10, 20)), "value": inputs[input]}]
        points = eval(input)

        vertices = []
        is_first = True
        for vertex in points:
            vertex_json = {}
            if is_first:
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
                is_first = False
            else:
                json_id = count
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
            vertex_json['x'] = vertex[0]
            vertex_json['y'] = vertex[1]
            vertices.append(vertex_json)
            count += 1
        sub_json_data['vertices'] = vertices
        data.append(sub_json_data)
    return json.dumps(data)


def send_to_rlef(img_path, model_id, tag,label, annotation = None, confidence_score=100, prediction='predicted'):
    print("Sending")
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    
    payload = {
        'model': model_id,
        'status': 'backlog',
        'csv': 'csv',
        'label': label,
        'tag': tag,
        'model_type': 'imageAnnotation',
        'prediction': prediction,
        'confidence_score': confidence_score
        # 'imageAnnotations': str(annotation)
    }
    files = [('resource', (f'{img_path}', open((img_path), 'rb'), 'image/png'))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    
    print('code: ', response.status_code)
    


if __name__ == '__main__':

    image_dir = 'text-seg-images'

    img_list = os.listdir(image_dir)





    count = 0
    
    exceptions = []
    threads = []

    flag = False
    # df = pd.read_csv('dataSetCollection_Sigma_Set_resources.csv')

    for idx, img_path in enumerate(img_list):
        # print(img_path, label_path)
        # img_path = os.path.join(image_dir, img_path)
        img_path = f"{image_dir}/{img_path}"

        ################ REMEMBER TO CHANGE THE MODEL ID ##################
        
        model_id = '6685900f97eae5e91291d0f4'
        tag = 'new-rack-india'
        label = 'object'
        img_count = 0
        
        threads.append(Thread(target = send_to_rlef, args = (img_path, model_id, tag, label )))
        
        if idx % MAX_THREADS==0:

            for th in threads:
                th.start()
            for th in threads:
                th.join() 

            threads = []

            # print(f'UPLOAD STATUS:{img_count}')
    
        img_count+=1
   
        
    if len(threads) > 0:
        for th in threads:
            th.start() 
        for th in threads:
            th.join() 
        threads = []

    print('Data Upload Done')