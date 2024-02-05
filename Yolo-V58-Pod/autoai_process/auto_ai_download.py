# Native Library Import
import os
import time
import json
from threading import Thread

# Pip or Installed Library Import
import yaml
import numpy as np
import pandas as pd
from google.cloud import storage
import cv2
# Custom file Import
from autoai_process import Config

BUCKET_NAME = Config.AUTOAI_BUCKET
MAX_NUMBER_THREADS = 100

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'faceopen_key.json'

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def get_files(folder_path):
    files = [f for f in os.listdir(folder_path)]
    files = sorted(files)
    return files


def download_blob(output_folder_path, blob_path, value):
    if Config.PROJECT_TYPE == "classification":
        filename = f"{(os.path.basename(blob_path)).split('.')[0]}.{Config.EXTENSION}"
        output_file_path = os.path.join(output_folder_path, value, filename)
    elif Config.PROJECT_TYPE == "detection" or Config.PROJECT_TYPE == "segmentation":
        filename = f"{value}"
        output_file_path = os.path.join(output_folder_path, filename)

    blob = bucket.blob('/'.join(blob_path.split('/')[3:]))

    blob.download_to_filename(output_file_path)


def download_files(input_csv_path, output_folder_path):
    print("input_csv_path-----------> ", input_csv_path)
    df = pd.read_csv(input_csv_path)
    gcp_paths = df['GCStorage_file_path']

    # if Config.PROJECT_TYPE == "classification":
    #     temp = df['label']
    #     for label in set(temp):
    #         if not os.path.exists(os.path.join(output_folder_path, label)):
    #             os.mkdir(os.path.join(output_folder_path, label))
    # elif Config.PROJECT_TYPE == "detection" or Config.PROJECT_TYPE == "segmentation":
    #     temp = df['name']

    temp = df['name']
    threads = []
    start = time.time()

    for idx, (blob_path, value) in enumerate(zip(gcp_paths, temp)):
        threads.append(Thread(target=download_blob,
                              args=(output_folder_path, blob_path, value)))
        if idx % MAX_NUMBER_THREADS == 0:
            for th in threads:
                th.start()
            for th in threads:
                th.join()

            print(f"Data Download Status : {idx}/{df.shape[0]}")
            threads = []

    if len(threads) > 0:
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        print(f"Data Download Status : {df.shape[0]}/{df.shape[0]}")

    print('Time taken : %s' % str(time.time() - start))


def csv_to_json_convert(csv_file, json_file_name, dataset_dir, single_cls):
    input_csv = csv_file

    images = []

    image_dict = {}

    annotations = []
    categories = []

    catagory_dict = {}

    print("input_csv------> ", input_csv)
    df = pd.read_csv(input_csv)
    categories_set = set()

    count = 0

    for index, row in (df.iterrows()):
        try:
            annotations_csv = row['imageAnnotations']
            annotations_csv = annotations_csv.replace("\'", "\"")
            objects = json.loads(str(annotations_csv))
            for object in objects:
                label = object['selectedOptions'][1]['value']
                label = label.lower()
                categories_set.add(label)
        except:
            continue

    Total_annotation = 0
    
    if single_cls == False:
        for i, label in enumerate(categories_set):
            catagory_dict[label] = i
            categories.append({
                "id": catagory_dict[label],
                "name": label,
                "supercategory": "none"
            })
    else:
        catagory_dict['obj'] = 0
        categories.append({
            "id": catagory_dict['obj'],
            "name": 'obj',
            "supercategory": "none"
        })

    annotation_count = 0

    for index, row in (df.iterrows()):

        annotations_csv = row['imageAnnotations']

        image_dict[row['name']] = len(image_dict.keys())

        if os.path.exists(os.path.join(os.path.join(dataset_dir, "images", row['name']))):
            img = cv2.imread(os.path.join(dataset_dir, "images", row['name']))
        else:
            img = cv2.imread(os.path.join(dataset_dir, row['name']))

        try:
            height, width, depth = img.shape
        except:
            continue
        image = {
            "id": image_dict[row['name']],
            "file_name": row['name'],
            "path": None,
            "width": width,
            "height": height,
            "depth": depth
        }
        images.append(image)

        # JSON does not read single quotes. So replacing single quotes with double
        try:
            annotations_csv = annotations_csv.replace("\'", "\"")
        except:
            continue

        objects = json.loads(str(annotations_csv))
        for object in objects:
            Total_annotation += 1
            # try:
            # print("Total_annotation-----> ", Total_annotation)
            # print("object----> ", object)

            label = object['selectedOptions'][1]['value'] if "selectedOptions" in object.keys() and isinstance(object['selectedOptions'], list) and object['selectedOptions'] else "Catheter-unkwown"

            if label == "Catheter-unkwown":
                print('\n', row['name'], '\n')
                from pprint import pprint
                pprint(object)
                print('\n')

            px = [vertex['x'] for vertex in object['vertices']]
            py = [vertex['y'] for vertex in object['vertices']]
            poly = [(vertex['x'], vertex['y'])
                    for vertex in object['vertices']]
            poly = [p for x in poly for p in x]
            
            if single_cls:
                label = 'obj'

            obj = {
                "id": annotation_count,
                "image_id": image_dict[row['name']],
                "bbox": [float(np.min(px)), float(np.min(py)), float(np.max(px)),
                         float(np.max(py))],
                "segmentation": [poly],
                "category_id": catagory_dict[label.lower()],
                "iscrowd": 0
            }
            annotation_count += 1
            annotations.append(obj)

    print("Total Annotations", Total_annotation)
    print("Registered Annotations", annotation_count)
    print('\n')

    json_file = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    out_file = open(json_file_name, "w")

    json.dump(json_file, out_file, indent=6)
    out_file.close()
    print(f"csv converted to json with name: {json_file_name}")


def json_to_txt_convert(json_file, training_id, dataset_dir, project_type, single_cls):
    print("Json conversion started")
    jsfile = json.load(open(json_file, "r"))

    image_id = {}

    for image in jsfile["images"]:
        image_id[image['id']] = image['file_name']

    for itr in range(len(jsfile["annotations"])):
        ann = jsfile["annotations"][itr]
        poly = ann["segmentation"][0]
        img = cv2.imread(dataset_dir + "/images/" + image_id[ann["image_id"]])
        try:
            height, width, depth = img.shape
        except:
            continue
        if project_type == 'detection':
            xmin = 999
            ymin = 999
            xmax = -1
            ymax = -1

            for i in range(len(poly) // 2):
                xmin = min(xmin, poly[2 * i])
                xmax = max(xmax, poly[2 * i])
                ymin = min(ymin, poly[2 * i + 1])
                ymax = max(ymax, poly[2 * i + 1])

            bbox = [ann["category_id"], (xmax + xmin) / (2 * width), (ymax + ymin) / (2 * height), (xmax - xmin) / width,
                    (ymax - ymin) / height]

        elif project_type == 'segmentation':
            bbox = [ann["category_id"]]
            for i in range(len(poly) // 2):
                _ = poly[2 * i] / width
                bbox.append(_)
                _ = poly[2 * i + 1] / height
                bbox.append(_)

        label_dir = os.path.join(dataset_dir, "labels")

        os.makedirs(label_dir, exist_ok=True)

        if os.path.exists(os.path.join(
                label_dir, os.path.splitext(os.path.basename(image_id[ann["image_id"]]))[0] + ".txt")):
            file = open(os.path.join(
                label_dir, os.path.splitext(os.path.basename(image_id[ann["image_id"]]))[0] + ".txt"), "a")
            file.write("\n")
            file.write(" ".join(map(str, bbox)))
        else:
            file = open(os.path.join(
                label_dir, os.path.splitext(os.path.basename(image_id[ann["image_id"]]))[0] + ".txt"), "a")
            file.write(" ".join(map(str, bbox)))
        file.close()

    classes = {i["id"]: i["name"] for i in jsfile["categories"]}

    yaml_file = {
        "train": f"{str(os.getcwd())}/datasets/{training_id}/images",
        "val": f"{str(os.getcwd())}/datasets/test_{training_id}/images"
    }
    if Config.single_cls:
        yaml_file["nc"] = 1
        yaml_file["names"] = {0: 'obj'}
    else:
        yaml_file["nc"] = len(classes)
        yaml_file["names"] = classes
    if Config.gray:
        yaml_file["ch"] = 1
    if Config.aug_col:
        hsv_h: 0.015
        hsv_s: 0.5
        hsv_v: 0.15
    yaml_file_path = os.path.join("datasets", f"{training_id}.yaml")
    file = open(yaml_file_path, "w")
    yaml.dump(yaml_file, file)
