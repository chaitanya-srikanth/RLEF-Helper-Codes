import requests, json
data = {
  "_id":  "66c478aa689210dc56bf0869",
  "model": "6685900f97eae5e91291d0f4",
  "created_at": "2024-08-20T11:06:18.568Z",
  "modelDescription": "Training with workstation-testing",
  "assigned_pod": "None",
  "status": "Queued",
  "resourcesFileName": "resources_66c478aa689210dc56bf0869.csv",
  "resourcesFileGCStoragePath": "gs://auto-ai_resources_fo/project_6682914be830ed90533d92b8/model_6685900f97eae5e91291d0f4/collections/modelCollections/modelCollection_66c478aa689210dc56bf0869/resources_66c478aa689210dc56bf0869.csv",
  "model_architecture": "dis-pod-client",
  "defaultDataSetCollectionId": "66c46700689210dc56bda21b",
  "defaultDataSetCollectionResourcesFileName": "defaultDataSetCollection_66c46700689210dc56bda21b_resources.csv",
  "modelArchitecture": "dis-pod-client",
  "createrMailId": "chaitanya.srikanth@techolution.com",
  "architectureIpAddress": "http://34.70.236.176:8501",
  "hyperParameter": {
    "project_type": "segmentation",
    "image_extention": "jpg",
    "epochs": 100,
    "batch": 8,
    "imgsz": 640,
    "gray": "false",
    "single_cls": "false",
    "aug_col": "true",
    "workers": 8,
    "device": [
      0
    ],
    "save_freq": 50,
    "integrity_confidence_threshold": 0.7,
    "patience": 2000
  },
  "defaultDataSetCollectionResourcesFileGCStoragePath": "gs://auto-ai_resources_fo/project_6682914be830ed90533d92b8/model_6685900f97eae5e91291d0f4/collections/modelCollections/modelCollection_66c478aa689210dc56bf0869/defaultDataSetCollection_66c46700689210dc56bda21b_resources.csv"
}
url = "http://34.70.236.176:8501/train"
response = requests.post(url,json = data)
print(response.text)