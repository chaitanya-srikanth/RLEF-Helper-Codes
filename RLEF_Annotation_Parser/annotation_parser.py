import pandas as pd 
import os 
import json
import requests 
import random 
import threading 
import cv2 

from resize_annotations import resize_polygon_annotations
import uuid 



def find_left_upper_right_down(points):
    if not points:
        return None, None  # Return None if the list is empty

    left_upper = [min(points, key=lambda point: point[0])[0], min(points, key=lambda point: point[1])[1]]
    right_down = [max(points, key=lambda point: point[0])[0], max(points, key=lambda point: point[1])[1]]

    return left_upper, right_down





MODEL_ID = "66efcedb79694c9cc31e3973"
FOLDER_NAME = 'text-seg-images'
MAX_THREAD = 40
df = pd.read_csv('dataSetCollection_testing-dataset-alpha_resources.csv')
# annotation_path = 'annotations'


# class_map = {
#     'barcode' : 'barcode', 
#     'ProductName' : 'product_name', 
#     'LotNo' : 'lot_no', 
#     'RefNo' : 'ref_no', 
#     'use_by' : 'use_by', 
#     'Qrcode' : 'delete', 
#     'mfg_date' : 'delete', 
#     'BarcodeNo' : 'delete'
# }


def pointPath_To_WeirdAutoaiAnnotationFormat(bboxes, label):
    li = {}
    # obj = "label"
    for bbox, lbl in zip(bboxes,label):
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

        li[f"[[{xmin}, {ymin}], [{xmin}, {ymax}], [{xmax}, {ymax}], [{xmax}, {ymin}]]"] = lbl   ## CHANGE IT LATER ACCORDINGLY

    rlef_format = json_creater(li, True)
    return rlef_format


def multipointPath_To_WeirdAutoaiAnnotationFormat(annotations, label):
    li = {}
    # obj = "label"
    for ann,lbl in zip(annotations, label):

        li[f"[[{ann[0][0]}, {ann[0][1]}], [{ann[1][0]}, {ann[1][1]}], [{ann[3][0]}, {ann[3][1]}], [{ann[2][0]}, {ann[2][1]}]]"] = lbl   ## CHANGE IT LATER ACCORDINGLY
    rlef_format = json_creater(li, True)

    return rlef_format

def segmentation_annotation_rlef(segments, label):
    li = {}

    for idx, segment in enumerate(segments):
        li[str(segment)] = label[idx]
    rlef_format = json_creater(li,True)
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
        'confidence_score': confidence_score,
        'imageAnnotations': str(annotation)
    }
    files = [('resource', (f'{img_path}', open((os.path.join(FOLDER_NAME, img_path)), 'rb'), 'image/png'))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # print(response.text)
    print('code: ', response.status_code)

count = 0

threads = []
for idx in range(len(df)):
    try:
        filenames = df['name'][idx].split('\\')
        if len(filenames) == 1:
            filename = filenames[0]
        else:
            filename = filenames[1]

        json_path = filename.replace('.png', '.json')
        tag = df['tag'][idx]
        label = df['label'][idx]
        # ann_file_path = os.path.join(annotation_path, json_path)

        try:
            dictionary = json.loads(df['imageAnnotations'][idx])
        except:
            try:
                os.remove(f'{FOLDER_NAME}/{filename}')
            except:
                print(f'{filename} has been removed')

        final_annotations = []
        image = cv2.imread(f'{FOLDER_NAME}/{filename}')
        height, width, _ = image.shape
        final_class = []
        
        for idx, entry in enumerate(dictionary):
            ### Read the image shape
            # print(filename)
        
            
            vertex_list = []
            clss_value = entry['selectedOptions'][1]['value']
            vertices = entry['vertices']
            for point in vertices:
                vertex_list.append([point['x'], point['y']])
            
            # left, right = find_left_upper_right_down(vertex_list)
            # if class_map[clss_value] != 'delete':
            final_annotations.append(vertex_list)
            final_class.append("text")




        # resized_image = cv2.resize(image, (640, 480), cv2.INTER_LINEAR)
        
        # resized_path = f'resized_images/{uuid.uuid1()}.png'
        # cv2.imwrite(resized_path, resized_image)
        # resized_annotations = resize_polygon_annotations((width, height), new_size, [vertex_list])


        # final_class = ['object'] * len(vertex_list)
        rlef_format = segmentation_annotation_rlef(final_annotations, final_class)
        # send_to_rlef(resized_path, MODEL_ID ,"US-Data", "aug-14",rlef_format)
        # send_to_rlef(filename, MODEL_ID ,tag,label,rlef_format)
        # break
        thread = threading.Thread(target = send_to_rlef, args =(filename, 
                                                            MODEL_ID ,tag,label,rlef_format,))

        
        threads.append(thread)
        
        if len(threads) % MAX_THREAD ==0:
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            count  +=  len(threads)
            print(f'STATUS : {count}/{len(df)}')
            threads = []
    except Exception as e:
        print(e)
        continue 
        
    
    
if len(threads) > 0:
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    threads = [] 

print('DONE')
