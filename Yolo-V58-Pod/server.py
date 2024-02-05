# Native Library Import
import sys
import json
import traceback
import threading
import shutil
import os

# Pip or Installed Library Import
from flask import Flask, request
from colorama import Fore, Style


# Custom file Import
from autoai_process import Config, gcp_train_utils
from autoai_process.builtin_func import auto_ai_connect
from autoai_process import train_test_process
import yolo_v8_assets

print((Fore.RED if Config.ENV != 'prod' else Fore.GREEN) +
      f'Pod Started in {Config.ENV} setting', Style.RESET_ALL)

Config.MODEL_STATUS = 'None'
Config.POD_STATUS = 'Available'

# Creating the app
app = Flask(__name__)


@app.get("/")
def root():
    print('ip_address', request.remote_addr)
    return {"server running": "true"}


@app.route('/')
def home():
    return "Hello, World! This is the Train-Pod. The pod is active", 200


@app.route('/test', methods=["POST"])
def test_model():
    try:
        print(Config.POD_STATUS, Config.MODEL_STATUS)
        if request.is_json:
            data = request.get_json()
            Config.MODEL_VERSION = int(data['startCheckpointFileArray'][0]['collectionHyperparameter'].get('model_version', 8))
            if Config.POD_STATUS == 'Available':
                Config.MODEL_STATUS = "None"
                p1 = threading.Thread(
                    target=train_test_process.start_test, args=(data,))
                p1.start()
                return "Testing started", 200
            else:
                return "Pod busy", 503
        else:
            return "Request must be JSON", 500
    except Exception:
        print("####################### ERROR #############################")
        print("Error while registering training pod :  ")
        traceback.print_exc(file=sys.stdout)
        print("####################### ERROR #############################")
        return ("Error : %s while queueing model" % 500)


@app.route('/train', methods=["POST"])
def train_model():
    try:
        print('ip_address', request.remote_addr)
        print(Config.POD_STATUS, Config.MODEL_STATUS)
        if request.is_json:
            data = request.get_json()
            print(data)
            # ANANDHA
            folder_path = "./runs"

            # ASHER
            # folder_path ="/usr/src/ultralytics/runs"

            try:
                # Refreshing the ultralytics training runs folder to remove previous results
                shutil.rmtree(folder_path) # Remove
                os.mkdir(folder_path) # Recreate
            except FileNotFoundError:
                print(f"The folder '{folder_path}' does not exist.")
            
            Config.aug_col = bool(data['hyperParameter'].get('aug_col', False))
            Config.gray = bool(data['hyperParameter'].get('gray', False))
            Config.single_cls = bool(data['hyperParameter'].get('single_cls', False))             
            Config.PROJECT_TYPE = str(data['hyperParameter'].get('project_type', "")).lower()
            Config.MODEL_VERSION = int(data['hyperParameter'].get('model_version', 8))
            Config.yolo_type = str(data['hyperParameter'].get('yolo_type', 'n'))
            if Config.MODEL_VERSION not in [5,8]:
                print("'model_version' should be either 5 or 8", 403)
                exit()

            if not Config.PROJECT_TYPE:
                print("Please set 'project_type' key in hyperParameter", 403)
                exit()

            if Config.PROJECT_TYPE not in ['segmentation', 'detection']:
                print("'project_type' should be either 'segmentation' or 'detection'", 403)
                exit()

            Config.EXTENSION = str(data['hyperParameter'].get('image_extention', "")).lower()
            if not Config.EXTENSION:
                print("Please set 'image_extention' key in hyperParameter", 403)
                exit()

            if Config.EXTENSION not in ['jpg', 'png']:
                print("'project_type' should be either 'jpg' or 'png'", 403)
                exit()

            if Config.MODEL_VERSION==5:
                Config.PROJECT_TYPE=='segmentation'
                Config.MODEL_TYPE = "yolov5s.pt"

            elif Config.MODEL_VERSION==8:
                if Config.yolo_type == 'n':
                    Config.MODEL_TYPE = "yolov8n-seg.pt" if Config.PROJECT_TYPE=='segmentation' else "yolov8n.pt"
                elif Config.yolo_type == 's':
                    Config.MODEL_TYPE = "yolov8s-seg.pt" if Config.PROJECT_TYPE=='segmentation' else "yolov8s.pt"
                elif Config.yolo_type == 'm':
                    Config.MODEL_TYPE = "yolov8m-seg.pt" if Config.PROJECT_TYPE=='segmentation' else "yolov8m.pt"
                shutil.copyfile(os.path.join(yolo_v8_assets.__path__[0], Config.MODEL_TYPE), os.path.join(str(os.getcwd()), Config.MODEL_TYPE))

            else:
                return "Unsupported model", 403
            
            if Config.POD_STATUS == 'Available':
                Config.MODEL_STATUS = "None"
                p1 = threading.Thread(
                    target=train_test_process.start_train, args=(data,))
                p1.start()
                return "Training started", 200
            else:
                return "Pod busy", 503
        else:
            return "Request must be JSON", 500
    except Exception as e:
        print("####################### ERROR #############################")
        print("Error while registering training pod :  ")
        traceback.print_exc(file=sys.stdout)
        print("####################### ERROR #############################")
        return "Error : %s while queueing model" % e, 500


@app.route('/get_update', methods=["GET"])
def get_update():
    try:
        if Config.MODEL_STATUS == 'Training Completed' or Config.MODEL_STATUS == "Testing Completed":
            print(Config.POD_STATUS, Config.MODEL_STATUS)
            Config.POD_STATUS = 'Available'
            Config.MODEL_STATUS = 'None'
            return {'pod_status': Config.POD_STATUS, "model_status": 'Training Completed'}, 200

        else:
            print(Config.POD_STATUS, Config.MODEL_STATUS)
            return {'pod_status': Config.POD_STATUS, 'model_status': Config.MODEL_STATUS}, 200
    except Exception as e:
        print("####################### ERROR #############################")
        print("Error while returning the pod and model status :  ", e)
        print("####################### ERROR #############################")
        return "Error : %s while queueing model" % e, 500


if __name__ == "__main__":
    # Do a Hard Reset
    # auto_ai_connect.reset()
    print('Ping at ->')
    os.system('curl ipinfo.io/ip ; echo')
    app.run("0.0.0.0", port=8501, debug=False)
