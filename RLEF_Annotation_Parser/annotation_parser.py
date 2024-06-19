import pandas as pd 
import os 
import json
import requests 
import random 
import threading 

def find_left_upper_right_down(points):
    if not points:
        return None, None  # Return None if the list is empty

    left_upper = [min(points, key=lambda point: point[0])[0], min(points, key=lambda point: point[1])[1]]
    right_down = [max(points, key=lambda point: point[0])[0], max(points, key=lambda point: point[1])[1]]

    return left_upper, right_down










# def pointPath_To_WeirdAutoaiAnnotationFormat(bboxes, label):
#     li = {}
#     # obj = "label"
#     for bbox, lbl in zip(bboxes,label):
#         xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

#         li[f"[[{xmin}, {ymin}], [{xmin}, {ymax}], [{xmax}, {ymax}], [{xmax}, {ymin}]]"] = lbl   ## CHANGE IT LATER ACCORDINGLY

#     rlef_format = json_creater(li, True)
#     return rlef_format


def multipointPath_To_WeirdAutoaiAnnotationFormat(annotations, label):
    li = {}
    # obj = "label"
    for ann,lbl in zip(annotations, label):

        li[f"[[{ann[0][0]}, {ann[0][1]}], [{ann[1][0]}, {ann[1][1]}], [{ann[3][0]}, {ann[3][1]}], [{ann[2][0]}, {ann[2][1]}]]"] = lbl   ## CHANGE IT LATER ACCORDINGLY
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
    files = [('resource', (f'{img_path}', open((img_path), 'rb'), 'image/png'))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # print(response.text)
    print('code: ', response.status_code)
    

df = pd.read_csv('dataSetCollection_training-20-classes_resources.csv')
annotation_path = 'annotations'



threads = []
for idx in range(len(df)):
    filenames = df['name'][idx].split('\\')
    if len(filenames) == 1:
        filename = filenames[0]
    else:
        filename = filenames[1]

    json_path = filename.replace('.png', '.json')
    ann_file_path = os.path.join(annotation_path, json_path)

    try:
        dictionary = json.loads(df['imageAnnotations'][idx])
    except:
        try:
            os.remove(f'images/{filename}')
        except:
            print(filename)

    final_annotations = []
    final_class = []
    for idx, entry in enumerate(dictionary):
        vertex_list = []
        clss_value = entry['selectedOptions'][1]['value']
        vertices = entry['vertices']
        for point in vertices:
            vertex_list.append([point['x'], point['y']])
        left, right = find_left_upper_right_down(vertex_list)
        final_annotations.append([left[0], left[1], right[0], right[1]])
        # final_annotations.append(vertex_list)
        final_class.append(clss_value)

 

    ### CUSTOMIZE FROM HERE ###
    try:
        if final_class[0] == final_class[1] == 'baylis':
            # continue
            rlef_format = pointPath_To_WeirdAutoaiAnnotationFormat(final_annotations, final_class)
            thread = threading.Thread(target = send_to_rlef, args =(f"training-images/{filename}", "661d03fa20cc192af059e8d1","training-image", "training-set",rlef_format,))
            threads.append(thread)
        else:
            rm_index = None 
            for idx,clss in enumerate(final_class):
                if clss == 'baylis':
                    rm_index = idx 

            final_class.pop(rm_index)
            final_annotations.pop(rm_index)
            rlef_format = pointPath_To_WeirdAutoaiAnnotationFormat(final_annotations, final_class)
            thread = threading.Thread(target = send_to_rlef, args =(f"training-images/{filename}", "661d03fa20cc192af059e8d1","training-image", "training-set",rlef_format,))
            threads.append(thread)
        if len(threads) > 20:
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            threads = []
            
            

    except Exception as e:
        continue

    
if len(threads) > 0:
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    threads = []