import os 
import cv2 
import numpy as np
import requests
import json
import random
import os
import numpy as np
import threading 


MAX_THREADS = 100


names = ["arducam", "baylis", "black-box", "wawa-orange", "wawa-violet", "wawa-yellow"]

def pointPath_To_WeirdAutoaiAnnotationFormat(bboxes, label):
    li = {}
    # obj = "label"
    idx = 0
    for bbox,lbl in zip(bboxes,label):
        # xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        annotation_str = "["
        for point in bbox:
            # print(point[0], point[1])

            annotation_str += f"[{point[0]}, {point[1]}],"
        
        annotation_str = annotation_str[:len(annotation_str)-1]
        annotation_str += "]"
        li[annotation_str] = names[int(lbl)]

    rlef_format = json_creater(li, True)
    return rlef_format


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


def send_to_rlef(img_path, model_id, tag,label, annotation, confidence_score=100, prediction='predicted'):
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
        'confidence_score': confidence_score,
        'imageAnnotations': str(annotation)
    }
    files = [('resource', (f'{img_path}', open((img_path), 'rb'), 'image/png'))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    
    print('code: ', response.status_code)
    

root_dir_path = "train/images"
root_label_path = 'train/labels'
image_list = os.listdir(root_label_path)
threads = []
for idx, path in enumerate(image_list):
    image_path = path.replace('.txt', '.jpg')
    image_path = os.path.join(root_dir_path, image_path)
    file_path = os.path.join(root_label_path, path) 


    with open(file_path, "r") as file:
        # Read all lines from the file
        lines = file.readlines()

    # Print the contents of the file


    all_annotations = []
    for line in lines:
        annotation_list = line.strip().split(' ') # strip() removes any leading/trailing whitespace and newline characters
        all_annotations.append(annotation_list)

    if len(all_annotations) == 0:
        continue
    # image_path = "6634d1e0a57229af76b87086\\images\\1a2ad332-093e-11ef-aa7c-42010a800043.jpg"


    
    image = cv2.imread(image_path)
    bounding_boxes = []
    classes = []
    for annotation in all_annotations:
        h,w,c = image.shape
        clss = annotation[0]
        points_float = annotation[1:]
        points_float = [float(x) for x in points_float]
 
        points_pixel = [(int(x * w), int(y * h)) for x, y in zip(points_float[::2], points_float[1::2])]
  
        bounding_boxes.append(points_pixel)
        classes.append(clss)
        # print('###########################')
   
    # print(classes)
    rlef_format = pointPath_To_WeirdAutoaiAnnotationFormat(bounding_boxes, classes)
    # print(names[int(classes[0])])
    # send_to_rlef(image_path, "6683b6359bae354d53f5bc0e", "rf-aug", "dummy", rlef_format)
    # break
 
    thread = threading.Thread(target = send_to_rlef, args = (image_path,"6683b6359bae354d53f5bc0e","rf-aug",names[int(classes[0])],rlef_format,))

    threads.append(thread)

    if len(threads) % 30 == 0:
        for th in threads:
            th.start() 
        for th in threads:
            th.join() 
        print(f" Files uploaded : {idx}/{len(image_list)}")
        threads = []
    


  
    
if len(threads) > 0:
    for th in threads:
        th.start() 
    for th in threads:
        th.join()