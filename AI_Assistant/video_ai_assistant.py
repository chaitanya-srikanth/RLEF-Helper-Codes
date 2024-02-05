import requests
from flask import Flask, request, jsonify
import traceback
from datetime import date
import os
from pprint import pprint
import numpy as np
import time
import uuid
import json
from PIL import Image
from dotenv import load_dotenv
import cv2
import requests
from ultralytics import YOLO
from results import process_image
import random
import math
import operator
from functools import reduce 

# Load environment variables from the .env file
load_dotenv()

model = YOLO('latest_barcode.pt')

def json_creater(inputs, closed):
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
        if len(points) > 0:
            center = tuple(
                map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
            sorted_coords = sorted(points, key=lambda coord: (-135 - math.degrees(
                math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        else:
            sorted_coords = []
        vertices = []
        is_first = True
        for vertex in sorted_coords:
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

def barcode_detect(img_path):
    results = model(img_path)
    # self.results = results
    # print('self.results----> ', self.results)    
    bboxes = []
    labels = []
    for result in results:
        labels.append(result.boxes.cls)
        bboxes.append(result.boxes.xyxy)
    labels = labels[0].tolist()
    bboxes = bboxes[0].tolist()
    if 1 in labels:
        return True
    else:
        return False

def send_image_annotation(model_id, tag, label, annotation, filename, csv):
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    payload = {
        'model': model_id,
        'status': 'backlog',
        'csv': f"{csv}",
        'label': label,
        'tag': tag,
        'model_type': 'imageAnnotation',
        'prediction': label,
        'confidence_score': '100',
        'imageAnnotations': str(annotation)
    }
    files = [('resource', (filename, open((filename), 'rb'), 'image/png'))]
    headers = {}
    requests.request("POST", url, headers=headers, data=payload, files=files)    
    
# # Sending file to RLEF.ai 
# def send_image_resource(file_name, csv, label, model='65a68738e2d7a27f782ed8f6'):
#     url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/"

#     payload = {
#         'model': model,
#         'status': 'backlog',
#         'csv': f"{csv}",
#         'label': label,
#         'tag': label,
#         'model_type': 'image',
#         'prediction': 'predicted',
#         'confidence_score': '100'
#     }
#     files = [
#         ('resource', (file_name , open(file_name, 'rb'), 'image/png'))
#     ]
#     headers = {}

#     response = requests.request(
#         "POST", url, headers=headers, data=payload, files=files)
    
#     # print(response.text)
    print("Image Sent")


def detect_blur_lap(gray_frame, thresh=100):
    # image = cv2.imread(image_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    
    # clss = 'blur' if variance <= thresh else 'sharp'
    return variance

# Variables
ALLOWED_EXTENSIONS = ['mp4', 'avi']

# Functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Creating a Flask server for listening for API calls
app = Flask(__name__)

@app.route('/')
def home():
    return "AI-Assistant server is available"


@app.route('/aihelp', methods=["POST"])
def make_prediction():
    try:
        file = request.files['resource']

        # Validating file
        if file and allowed_file(file.filename):
            # Saving the file with unique name
            file_id = uuid.uuid1().hex
            filename = file_id + '.' + file.filename.split('.')[-1]
            file.save(filename)
        else:
            return 'Invalid file type', 500        
        
        # generating unique id
        unique_id = uuid.uuid1().hex

        # mask_full_image_path = filename
        out_path = f"{unique_id}.png"
        frame_path = "frame.png"

        # image_path = "eb66f578-ae46-11ee-bb5c-a9f23fa16de4.png"
        thresh = 100
        cap = cv2.VideoCapture(filename)
        count = 0
        result = False
        best_frame = None
        best_frame_no = 0
        best_var = 0
        while True:
            ret, frame = cap.read()
            # print(frame.shape)
            if not ret:
                break
            count += 1
            cv2.imwrite(frame_path, frame)
            if barcode_detect(frame_path):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                var = detect_blur_lap(gray, thresh)
                if var > thresh:
                    result = True
                if var > best_var:
                    best_var = var
                    best_frame = frame.copy()
                    best_frame_no = count
                
        cap.release()
        
        main_dic = {
            "Number of Frames" : count,
            "Best_frame_number" : best_frame_no,
            "Highest_Lap" : best_var,
            "Result" : "Pass" if result else "Fail"   
        }
        
        if result:
            cv2.imwrite(out_path, best_frame)
            output = process_image(out_path)
            li = {}
            for out in output:
                xmin, ymin, xmax, ymax = out['xyxy']
                li[f"[[{xmin}, {ymin}], [{xmin}, {ymax}], [{xmax}, {ymax}], [{xmax}, {ymin}]]"] = out['type']
            
            annotations = json_creater(li, True)
            
            main_dic["Output"] = output
            # send_image_resource(file_name = out_path, csv = main_dic, label = "sharp")
            send_image_annotation(model_id = "6565b30a8ac019ca9eec77b7", tag = "laplace", label= "sharp", annotation= annotations, filename = out_path, csv= main_dic)
            
        
        response_packet = {
            'csv': f"{main_dic}"
        }
        # Clean up 
        # os.remove(mask_cropped_image_path)
        os.remove(frame_path)
        os.remove(filename)
        if result:
            os.remove(out_path)

        # Getting prediction
        return jsonify(response_packet), 200
    except Exception as e:
        print('Detailed Error')
        print(traceback.format_exc())
        return "Error : %s" % e, 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8506, debug=True)


# no of frames
# highest lap
# best_frame
# result