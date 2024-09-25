import os
from google.cloud import storage
from threading import Thread
import time
import pandas as pd
from datetime import datetime, date
import requests

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'faceopen_key.json'
storage_client = storage.Client()
bucket_name = 'auto-ai_resources_fo'
bucket = storage_client.bucket(bucket_name)

MAX_NUMBER_THREADS = 100

def download_blob(output_file_path, blob_path, value):
    # filename = f"{(os.path.basename(blob_path)).split('.')[0]}.jpg"
    # output_file_path = os.path.join(output_file_path, os.path.basename(value)) #LINUX SYSTEM
    output_file_path = f"{output_file_path}/{os.path.basename(value)}"
    blob = bucket.blob('/'.join(blob_path.split('/')[3:]))
    blob.download_to_filename(output_file_path)

def download_files(input_csv_path, output_folder_path):
    df = pd.read_csv(input_csv_path)
    gcp_paths = df['GCStorage_file_path']
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

def send_to_rlef(img_path, annotation, model_id, tag, confidence_score=100, label='RGB', prediction='predicted'):
    print("Sending")
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    # annotation = json_creater(annotation, True)
    payload = {
        'model': model_id,
        'status': 'backlog',
        'csv': 'csv',
        'label': 'RGB',
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


input_csv_path='dataSetCollection_testing-dataset-alpha_resources.csv'
output_folder_path='text-seg-images'
if os.path.exists(output_folder_path) is False:
    os.mkdir(output_folder_path)
download_files(input_csv_path, output_folder_path)


