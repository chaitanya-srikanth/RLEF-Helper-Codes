from autoai_process import Config
from ultralytics import YOLO
import yaml
from pprint import pprint


def main(yaml_path, epochs, weight_path, imgsz=640, model_type=Config.MODEL_TYPE, hyperparameter=dict()):
    """
    dataset_path: the path where training data is stored
    models_path: the path where the weights of the trained model needs to be stored
    frezze_layers: number of layers to freeze
    retrain_weight: path at which weights of the trained model are downloaded
    hyperparameter: parameters whose value is used to control the learning process

    The main objective of this function is to train the model on the data present in the dataset_path and
    store the weights of the trained model at the path "models_path".
    """
    batch = 16
    imgsz = 640
    epochs = 300
    workers = 8
    single_cls = False
    patience = 50
    device=[0]
    acc_dict = dict()
    if "epochs" in hyperparameter:
        epochs = hyperparameter["epochs"]
    if "imgsz" in hyperparameter:
        imgsz = hyperparameter["imgsz"]
    if "batch" in hyperparameter:
        batch = hyperparameter["batch"]
    if "single_cls" in hyperparameter:
        single_cls = hyperparameter["single_cls"]
    if "workers" in hyperparameter:
        workers = hyperparameter["workers"]
    if "patience" in hyperparameter:
        patience = hyperparameter["patience"]
    if "device" in hyperparameter:
        device = hyperparameter["device"]
    
    
    print('\n\n################################### hyperparameter ###################################')
    pprint(hyperparameter)
    print('################################### hyperparameter ###################################\n\n')

    # If segmentation: use mr= metrics/recall(M) and mp = metrics/precision(M)
    # If detection: use mr= metrics/recall(B) and mp = metrics/precision(B)
    # And then same formula as yolov5
    if Config.MODEL_VERSION == 8:
        model = YOLO(model_type)  # load a pretrained model (recommended for training)
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            single_cls=single_cls,
            patience=patience,
            device=device
            # device=hyperparameter['device']
        )
        # Not testing the model in this location
        # ret = model.val(data=yaml_path)
        # results = ret.results_dict
        #
        # if Config.PROJECT_TYPE == "segmentation":
        #     mp, mr = results['metrics/precision(M)'], results['metrics/recall(M)']
        #
        # elif Config.PROJECT_TYPE == "detection":
        #     mp, mr = results['metrics/precision(B)'], results['metrics/recall(B)']
        #
        # else:
        #     mp, mr = None, None
        #
        # acc_dict = dict()
        # data_file = open(yaml_path, 'r')
        # data_deets = yaml.safe_load(data_file)
        #
        # try:
        #     acc_dict['Total'] = round(2*(mr*mp)/(mp+mr), 5)
        # except ZeroDivisionError:
        #     acc_dict['Total'] = 0.0
        #
        # return acc_dict

    elif Config.MODEL_VERSION==5:
        train.run(imgsz=imgsz, data=yaml_path, weights = model_type, epochs=epochs, workers=0, **{"--batch-size": batch})
        test_list = val.run(data=yaml_path, weights = weight_path, workers=0)
        mr, mp, map50, map, loss1, loss2, loss3 = test_list[0]
        maps = test_list[1]
        maps50 = test_list[2]
        acc_dict = dict()
        data_file = open(yaml_path, 'r')
        data_deets = yaml.safe_load(data_file)
        for num, _ in enumerate(maps):
            temp_dict = dict()
            temp_dict['map50'] = maps50[num]
            temp_dict['map'] = _
            acc_dict[data_deets['names'][num]] = temp_dict
        acc_dict['Total'] = round(2*(mr*mp)/(mp+mr), 5)
        return acc_dict
        # val.run(imgsz=imgsz, data='coco128.yaml', weights='yolov5s.pt')
        # detect.run(imgsz=imgsz)
        # export.run(imgsz=imgsz, weights='yolov5s.pt')

    # elif Config.MODEL_VERSION==6:

    #     register_coco_instances('hand_dataset', {}, Config.TRAIN_JSON_PATH, Config.TRAIN_DATASET_PATH)


    #     cfg = get_cfg()

    #     cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    #     cfg.DATASETS.TRAIN = ("hand_dataset",)
    #     cfg.DATASETS.TEST = ()
    #     cfg.DATALOADER.NUM_WORKERS = 2
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    #     cfg.SOLVER.IMS_PER_BATCH = 2
    #     cfg.MODEL.DEVICE='cpu'

        # cfg.INPUT.MAX_SIZE_TRAIN: 720
        # cfg.INPUT.MIN_SIZE_TRAIN: 450
        # cfg.INPUT.MAX_SIZE_TEST: 2048
        # cfg.SOLVER.NUM_GPUS = 1
        # TOTAL_NUM_IMAGES = 138
        # single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
        # iterations_for_one_epoch = TOTAL_NUM_IMAGES / single_iteration

        # cfg.SOLVER.BASE_LR = 0.00025
        # cfg.SOLVER.MAX_ITER = 500

        # cfg.SOLVER.STEPS = []
        # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        # trainer = DefaultTrainer(cfg)
        # trainer.resume_or_load(resume=False)
        # trainer.train()

