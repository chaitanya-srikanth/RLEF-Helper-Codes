# train_func
import json
import os
import shutil
import sys
import traceback
import cv2
import pprint
# Pip or Installed Library Import
from colorama import Fore, Style
from glob import glob
# Custom file Import
from autoai_process import Config
from autoai_process.auto_ai_download import csv_to_json_convert, json_to_txt_convert, download_files
from autoai_process.builtin_func import auto_ai_connect
from autoai_process.gcp_train_utils import data_preprocessing
from training import test_model, train_model


class training_function():
    def __init__(self, model_details, retrain):
        self.model_details = model_details
        self.retrain = retrain

        # Paths can be updated, added, or removed based on developer requirements #

        # This is where training data would be stored
        self.parent_dir = os.path.join(
            Config.DATA_PATH, self.model_details['_id'])
        self.test_parent_dir = os.path.join(
            Config.DATA_PATH, f"test_{self.model_details['_id']}")
        self.dataset_path = os.path.join(
            Config.DATA_PATH, self.model_details['_id'], "images")
        # This is where the weights of the trained model would be stored
        if Config.MODEL_VERSION == 8:
            model_type = "segment" if Config.PROJECT_TYPE == "segmentation" else "detect"

            # ASHER
            # self.models_path = f"/usr/src/ultralytics/runs/{model_type}/train/weights/best.pt"

            # ANANDHA
            self.models_path = os.path.join("runs", model_type, "train", "weights", "best.pt")

        elif Config.MODEL_VERSION == 5:
            # model_type = "segment" if Config.PROJECT_TYPE=="segmentation" else "detect"

            # ASHER
            # self.models_path = f"/usr/src/ultralytics/runs/train/exp/weights/best.pt"

            # ANANDHA
            self.models_path = os.path.join("runs", "train", "exp", "weights", "best.pt")

        else:
            self.models_path = None

        if self.retrain:
            self.checkpoint_model_path = os.path.join(Config.MODELS_PATH, self.model_details['_id'],
                                                      "checkpoint", self.model_details['startCheckpointFileName'])
        print("Config.MODEL_VERSION------------> ", Config.MODEL_VERSION)
        print("self.models_path------------> ", self.models_path)
        # exit()
        # Path to csv file which contains info about the training data
        self.train_csv_path = os.path.join(
            Config.DATA_PATH, self.model_details['_id'], self.model_details['resourcesFileName'])

        # This is where testing data would be stored
        self.test_dataset_path = os.path.join(
            Config.DATA_PATH, f"test_{self.model_details['_id']}", "images")
        # Path to csv file which contains info about the testing data
        self.test_csv_path = os.path.join(
            Config.DATA_PATH, f"test_{self.model_details['_id']}",
            self.model_details['defaultDataSetCollectionResourcesFileName'])

        if Config.PROJECT_TYPE == "detection" or Config.PROJECT_TYPE == "segmentation":
            self.train_json_path = os.path.join(
                Config.DATA_PATH, self.model_details['_id'], f'{self.model_details["_id"]}.json')

            self.test_json_path = os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}",
                                               f"{self.model_details['_id']}.json")

            self.yaml_path = os.path.join("datasets", f'{self.model_details["_id"]}.yaml')

    def train(self):
        print("============== model_details ==============")
        print(self.model_details)

        Config.MODEL_STATUS = 'Downloading Data'

        # Downloading the data from the csv file for training and testing
        if Config.PROJECT_TYPE in ["detection", "segmentation", "classification"]:
            download_files(self.train_csv_path, self.dataset_path)
            download_files(self.test_csv_path, self.test_dataset_path)

        Config.MODEL_STATUS = 'Preprocessing Data'
        Config.TRAIN_DATASET_PATH = self.dataset_path

        if Config.PROJECT_TYPE == "detection" or Config.PROJECT_TYPE == "segmentation":
            # Annotation File creation CSV to JSON(COCO) for training and testing
            print("\n\nPreprocessing\n\n")

            csv_to_json_convert(self.train_csv_path, self.train_json_path, self.parent_dir)
            csv_to_json_convert(self.test_csv_path, self.test_json_path, self.test_parent_dir)

            json_to_txt_convert(self.train_json_path, self.model_details['_id'], self.parent_dir, Config.PROJECT_TYPE)
            json_to_txt_convert(self.test_json_path, self.model_details['_id'], self.test_parent_dir,
                                Config.PROJECT_TYPE)

        Config.MODEL_STATUS = "Training"
        Config.TEST_JSON_PATH = self.test_json_path
        Config.TRAIN_JSON_PATH = self.train_json_path

        # getting rid of previous trains
        # Clearin out the Val dir before starting the other 
        for dir in glob('runs/detect/val*'):
            shutil.rmtree(dir)

        print('Training started')

        ######################Singleclass######################
        if Config.single_cls:
            path = self.parent_dir + '/labels/'
            for _ in os.listdir(path):
                if _[-3:] == 'txt':
                    file_label = path + _
                    f = open(file_label, 'r')
                    cont = f.read()
                    cont = cont.split("\n")
                    f.close()
                    corrected = []
                    for __ in cont:
                        corrected.append('0' + __[1:])
                    corrected_str = "\n".join(corrected)
                    f = open(file_label, 'w')
                    f.write(corrected_str)
                    f.close()
                else:
                    try:
                        os.remove(path + _)
                    except:
                        shutil.rmtree(path + _)

            path = self.test_parent_dir + '/labels/'
            for _ in os.listdir(path):
                if _[-3:] == 'txt':
                    file_label = path + _
                    f = open(file_label, 'r')
                    cont = f.read()
                    cont = cont.split("\n")
                    f.close()
                    corrected = []
                    for __ in cont:
                        corrected.append('0' + __[1:])
                    corrected_str = "\n".join(corrected)
                    f = open(file_label, 'w')
                    f.write(corrected_str)
                    f.close()
                else:
                    try:
                        os.remove(path + _)
                    except:
                        shutil.rmtree(path + _)
        ##########################Gray#############################
        if Config.gray:
            path = self.parent_dir + '/images/'
            for _ in os.listdir(path):
                if _[-3:] == 'png':
                    img = cv2.imread(path + _)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(path + _, gray)
                else:
                    try:
                        os.remove(path + _)
                    except:
                        shutil.rmtree(path + _)

            path = self.test_parent_dir + '/images/'
            for _ in os.listdir(path):
                if _[-3:] == 'png':
                    img = cv2.imread(path + _)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(path + _, gray)
                else:
                    try:
                        os.remove(path + _)
                    except:
                        shutil.rmtree(path + _)

        # print("deets ----------->", self.model_details)
        # The call to start the training of the model should be made here #
        train_model.main(yaml_path=self.yaml_path,
                         epochs=1,
                         weight_path=self.models_path,
                         imgsz=640,
                         model_type=Config.MODEL_TYPE if not self.retrain else self.checkpoint_model_path,
                         hyperparameter=self.model_details['hyperParameter'])
        files_to_send = {
            'parentfile': [],
            'modelfile': [],
            'analyticfile': [],
            'additionalfile': []
        }

        files_to_send['parentfile'].append(self.models_path)
        files_to_send['modelfile'].append(self.models_path)
        print('Training Completed')

        print("Testing Started")
        Config.MODEL_STATUS = 'Testing'

        # =============== Starting the test ===============
        test_detail = dict()

        if 'defaultDataSetCollectionResourcesFileGCStoragePath' in self.model_details.keys():
            test_detail['defaultDataSetCollectionId'] = self.model_details['defaultDataSetCollectionId']
            test_detail['defaultDataSetCollectionFileName'] = os.path.basename(
                self.model_details['defaultDataSetCollectionResourcesFileGCStoragePath'])

            try:
                # This is where the predictions of the model will be stored
                # Model Analytics for the test data
                output_file = os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}",
                                           f"test_{self.model_details['_id']}.csv")
                statistics_file = os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}",
                                               "Segmentation_Statistics.json")

                acc = test_model.main(test_csv_path=self.test_csv_path,
                                      test_json_path=self.test_json_path,
                                      dataset_path=self.test_dataset_path,
                                      models_path=self.models_path,
                                      output_file=output_file,
                                      statistics_file=statistics_file,
                                      hyperparameters=self.model_details['hyperParameter'])

                # This is actually the F1 Score
                test_detail['accuracy'] = acc['Total']

                # Adding the test JSON
                with open(os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}",
                                       f"test_{self.model_details['_id']}.json"), "w") as outfile:
                    json.dump(acc, outfile)

                files_to_send['analyticfile'].append(os.path.join(Config.DATA_PATH,
                                                                  f"test_{self.model_details['_id']}",
                                                                  f"test_{self.model_details['_id']}.json"
                                                                  )
                                                     )

                files_to_send['analyticfile'].append(output_file)
                files_to_send['analyticfile'].append(statistics_file)

            except Exception:
                print(Fore.RED + "Test Failed")
                traceback.print_exc(file=sys.stdout)
                print(Style.RESET_ALL)

        # =============== Testing Completed =============== #

        Config.MODEL_STATUS = 'Uploading Files'

        # Sending trained data to AutoAI
        if Config.ENV != 'dev':
            auto_ai_connect.autoai_upload_files(
                url=Config.AUTOAI_URL_SEND_FILES,
                files=files_to_send,
                isDefault=True,
                id=self.model_details['_id'],
                test_detail=test_detail
            )

        Config.MODEL_STATUS = 'Training Completed'
        Config.POD_STATUS = "Available"
        if Config.MODEL_VERSION == 8:
            os.remove(Config.MODEL_TYPE)
        Config.MODEL_TYPE = None

        print(Fore.GREEN + "Training Completed")
        print(Style.RESET_ALL)
