# Native Library Import
import os
import json
import sys
import traceback
import yaml
from glob import glob
import shutil

# Pip or Installed Library Import
from colorama import Fore, Style


# Custom file Import
from autoai_process import Config
from training import test_model
from autoai_process.auto_ai_download import download_files, csv_to_json_convert
from autoai_process.builtin_func import auto_ai_connect
from autoai_process.gcp_train_utils import data_preprocessing

def json_to_txt_convert(json_file, training_id, dataset_dir, project_type):
    print("Json conversion started")
    jsfile = json.load(open(json_file, "r"))

    image_id = {}

    for image in jsfile["images"]:
        image_id[image['id']] = image['file_name']

    for ann in jsfile["annotations"]:
        poly = ann["segmentation"][0]
        
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

            bbox = [ann["category_id"], (xmax + xmin) / (2 * 640), (ymax + ymin) / (2 * 480), (xmax - xmin) / 640,
                  (ymax - ymin) / 480]

        elif project_type == 'segmentation':
            bbox = [ann["category_id"]]
            for i in range(len(poly) // 2):
                _ = poly[2 * i]/640.0
                bbox.append(_)
                _ = poly[2 * i+ 1]/480.0
                bbox.append(_)
              
        label_dir = os.path.join(dataset_dir, "labels")
        
        os.makedirs(label_dir,exist_ok = True)

        file = open(os.path.join(
            label_dir, image_id[ann["image_id"]][:-4] + ".txt"), "a")

        file.write(" ".join(map(str, bbox)) + "\n")
        file.close()

    print('jsfile["categories"]------------->\n  ', jsfile["categories"])
    classes = {i["id"] : i["name"] for i in jsfile["categories"]}
    print("classes -------------->", classes)

    yaml_file = {
      "train": f"{str(os.getcwd())}/datasets/{training_id}/images",
      "val": f"{str(os.getcwd())}/datasets/{training_id}/images",
  
      "nc": len(classes),

      "names": classes
    }

    yaml_file_path = os.path.join("datasets",f"{training_id}.yaml")
    file = open(yaml_file_path, "w")
    yaml.dump(yaml_file, file)

class testing_function():
    def __init__(self, model_details):
        self.model_details = model_details

        # Paths can be updated, added, or removed based on developer requirements #

        # This is where testing data would be stored
        self.test_parent_dir = os.path.join(
            Config.DATA_PATH, self.model_details["_id"])
        self.test_dataset_path = os.path.join(
            Config.DATA_PATH, self.model_details["_id"], "images")
        # Path to csv file which contains info about the testing data
        self.test_csv_path = os.path.join(Config.DATA_PATH, self.model_details['_id'], self.model_details['defaultDataSetCollectionResourcesFileName'])
        self.yaml_path = os.path.join("datasets",f'{self.model_details["_id"]}.yaml')

    def test(self):
        print("============== model_details ==============")
        print(self.model_details)

        Config.MODEL_STATUS = 'Downloading Data'

        # Downloading the data from the csv file
        download_files(self.test_csv_path, self.test_dataset_path)

        # print(Fore.RED + "From training/test_func.py Line 51")
        # print(Fore.RED + "Data Preprocessing step, if required has to be created in the file autoai_process/gcp_train_utils.py")

        data_preprocessing()


        print("Testing Started")
        Config.MODEL_STATUS = 'Testing'

        print(Fore.RED + "From training/test_func.py Line 61")
        print(Fore.RED + "The main() function present in the file training/test_model.py has to be updated based on developer requirements")

        # Clearin out the Val dir before starting the other 
        for dir in glob('runs/detect/val*'):
            shutil.rmtree(dir)

        # Iterating Through all models and running the test
        for val_index, checkpoint in enumerate(self.model_details["startCheckpointFileArray"]):

            files_to_send = list()

            try:
                # This is where the predictions of the model will be stored
                # Path to the weights of a particular checkpoint model
                weight_path = os.path.join(
                    Config.MODELS_PATH,
                    checkpoint['_id'],
                    'best.pt'
                )
                test_json_path = os.path.join(Config.DATA_PATH, f"{self.model_details['_id']}", f"{self.model_details['_id']}.json"
                    )
                output_dir = os.path.join(
                        Config.DATA_PATH,
                        checkpoint['_id']
                    )
                val_json_file = os.path.join(
                        Config.DATA_PATH,
                        self.model_details['_id'],
                        self.model_details['_id'] + '.json'
                    )
                # test_detail = dict()
                # test_detail['defaultDataSetCollectionId'] = self.model_details['defaultDataSetCollectionId']
                # test_detail['defaultDataSetCollectionFileName'] = os.path.basename(
                #     self.model_details['defaultDataSetCollectionResourcesFileGCStoragePath']
                # )
                csv_to_json_convert(self.test_csv_path, test_json_path, self.test_dataset_path)
                json_to_txt_convert(test_json_path,  self.model_details['_id'], self.test_parent_dir, Config.PROJECT_TYPE)
                # csv_to_json_convert(self.test_csv_path, train_json_file)
                # The call to start the testing of the model should be made here #

                output_file = os.path.join(Config.DATA_PATH, checkpoint["_id"],
                                           f"test_{checkpoint['_id']}.csv")
                statistics_file = os.path.join(Config.DATA_PATH, checkpoint["_id"],
                                               "Segmentation_Statistics.json")

                acc = test_model.main(test_csv_path=self.test_csv_path,
                                      test_json_path=test_json_path,
                                      dataset_path=self.test_dataset_path,
                                      models_path=weight_path,
                                      output_file=output_file,
                                      statistics_file=statistics_file,
                                      hyperparameters=checkpoint['collectionHyperparameter'])


                with open(os.path.join(Config.DATA_PATH, checkpoint["_id"], f"test_{checkpoint['_id']}.json"),
                          "w") as outfile:
                    json.dump(acc, outfile)

                files_to_send.append(
                    os.path.join(
                        Config.DATA_PATH, checkpoint["_id"], f"test_{checkpoint['_id']}.json")
                )

                files_to_send.append(os.path.join(Config.DATA_PATH, checkpoint["_id"],
                                           f"test_{checkpoint['_id']}.csv"))

                files_to_send.append(os.path.join(Config.DATA_PATH, checkpoint["_id"],
                                               "Segmentation_Statistics.json"))

                Config.MODEL_STATUS = 'Uploading Files'

                # Sending files to AutoAI
                print("files_to_send--------> \n", files_to_send)
                auto_ai_connect(self.model_details["_id"]).autoai_upload_files_test(
                    url=Config.AUTOAI_URL_SEND_FILES, 
                    files=files_to_send, 
                    parentCheckpointFileId=checkpoint["startCheckpointId"], 
                    accuracy=acc['Total'])

            except Exception as e:
                print(Fore.RED + "Test Failed", checkpoint['_id'])
                traceback.print_exc(file=sys.stdout)
                print(Style.RESET_ALL)

        Config.MODEL_STATUS = "Testing Completed"
        Config.POD_STATUS = "Available"

        print(Fore.GREEN + "Testing Completed")
        print(Style.RESET_ALL)
