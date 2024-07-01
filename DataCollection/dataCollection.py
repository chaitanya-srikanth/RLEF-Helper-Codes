import cv2
import os  
import shutil 
import uuid 
from concurrent.futures import ThreadPoolExecutor

import requests 


# names = ["wawa-darkgreen", "wawa-red", "wawa-darkblue", "wawa-orange", "wawa-yellow", "wawa-violet", 
#          "wawa-white", "wawa-darkgreen", "wawa-cyan", "wawa-gray", "wawa-brown", "meritmak", "guidezilla", 
#          "pilot", "baylis", "bardex", "phonebox", "lays", "tedhe-medhe", "cardbox"]

names = ["object"]
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
    if response.status_code != 200:
        print(response.text)
    print('code: ', response.status_code)
    


executor = ThreadPoolExecutor(max_workers=5)

IMG_DIR = 'frames'
model_id = "666ade1e38b32dfe3b13e213"
if os.path.exists(IMG_DIR) is False:
    os.mkdir(IMG_DIR)


cap = cv2.VideoCapture(1)
global label 
label = names[0]
while True:
    ret, frame = cap.read()
    if ret is False:
        break 
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break 
    # elif key == ord('l'):
    #     lbl_index+=1 
    #     label = names[lbl_index]
    

    elif key == ord('c'):
        image_path = os.path.join(IMG_DIR, f"{uuid.uuid1()}.jpg")
        cv2.imwrite(image_path, frame)
        print(image_path)
        tag = "Live-Stream"
        print(label)
        # send_to_rlef(image_path, model_id, tag, label)
        executor.submit(send_to_rlef, image_path, model_id, tag, label)

    cv2.imshow("Live Stream", frame)
cap.release()
cv2.destroyAllWindows()