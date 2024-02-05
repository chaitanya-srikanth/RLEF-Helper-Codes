import os, sys
from autoai_process import Config
import json
from ultralytics import YOLO
import pandas as pd
from sklearn.metrics import auc
import random
import string
from pprint import pprint

def calculate_iou(boxA, boxB):
    # Calculate the intersection over union (IoU) between two bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def nms(predicted_boxes, predicted_labels, confidence_scores, iou_threshold=0.5):
    # Zip boxes, labels, and scores into a single list
    detections = list(zip(predicted_boxes, predicted_labels, confidence_scores))

    # Sort detections by confidence score in descending order
    detections.sort(key=lambda x: x[2], reverse=True)

    # Final list of detections after NMS
    final_detections = []

    while detections:
        # Select the detection with the highest confidence and remove it from the list
        current_detection = detections.pop(0)
        final_detections.append(current_detection)

        # Compute IoU of the current detection with all the other detections
        detections = [detection for detection in detections if
                      not calculate_iou(current_detection[0], detection[0]) > iou_threshold]

    # Unzip the final detections back into separate lists
    final_boxes, final_labels, final_scores = zip(*final_detections)
    return list(final_boxes), list(final_labels), list(final_scores)


def generate_random_id(digit_count):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(digit_count))


def calculate_aps(total_ground_truths, detections):
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    print("sorted_detections", sorted_detections)

    aps = {}
    for class_id in total_ground_truths.keys():
        class_detections = [d for d in sorted_detections if d['class_id'] == class_id]
        tp = 0
        fp = 0
        precisions = [1.0]  # start with precision at 1
        recalls = [0.0]  # start with recall at 0
        for detection_index, detection in enumerate(class_detections):
            if detection['is_true_positive']:
                tp += 1
            else:
                fp += 1
            precision = tp / (tp + fp)
            recall = tp / total_ground_truths[class_id]  # fixed FN calculation
            precisions.append(precision)
            recalls.append(recall)

            if recall > 1:
                print("detection_index", detection_index)
                print("detection", detection)
                print("tp", tp)
                print("fp", fp)
                print("precision", precision)
                print("recall", recall)
                print(total_ground_truths[class_id])

        # Ensure precisions are monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Append the end point (recall at 1.0, precision at 0.0) if not already there
        if recalls[-1] != 1.0:
            recalls.append(1.0)
            precisions.append(0.0)

        # print("precisions", precisions)
        # print("recalls", recalls)
        
        for i in range(len(recalls)):
            recalls[i] = min(1.0, recalls[i])

        ap = auc(recalls, precisions)
        aps[class_id] = ap

    return aps


def convert_bounding_boxes_to_image_annotations(masks, confidence_scores, labels, is_closed=False, is_bb=False):
    formatted_data = []
    high_contrasting_colors = ['rgba(0,255,81,1)', 'rgba(255,219,0,1)', 'rgba(255,0,0,1)', 'rgba(0,4,255,1)',
                               'rgba(227,0,255,1)']

    # If the bounding box is in the format [x1, y1, x2, y2]
    # We need to convert it into [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    if is_bb:
        bb_masks = []
        for bb in masks:
            bb_masks.append([[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]])
        masks = bb_masks

    for mask_points_list, confidence_score, label in zip(masks, confidence_scores, labels):

        random_id_first = generate_random_id(8)
        vertices = []

        index = 0
        for mask_index, mask_points in enumerate(mask_points_list):

            # # Only using every 10th point to reduce the number of points
            # if mask_index % 30 != 0:
            #     continue

            x = mask_points[0]
            y = mask_points[1]

            if index == 0:
                vertex_id = random_id_first
            else:
                vertex_id = generate_random_id(8)

            vertex = {
                "id": vertex_id,
                "name": vertex_id,
                "x": int(x),
                "y": int(y),
            }
            vertices.append(vertex)

            index = index + 1

        mask_data = {
            "id": random_id_first,
            "name": random_id_first,
            "color": random.choice(high_contrasting_colors),
            "isClosed": is_closed,
            "vertices": vertices,
            "confidenceScore": int(confidence_score * 100),
            "selectedOptions": [{
                "id": "0",
                "value": "root"
            },
                {
                    "id": random_id_first,
                    "value": label
                }]
        }

        formatted_data.append(mask_data)

    return formatted_data


def main(test_csv_path, test_json_path, dataset_path, models_path, output_file, statistics_file, hyperparameters):
    test_csv = pd.read_csv(test_csv_path)
    
    print('Hyperparameter ------------- >')
    pprint(hyperparameters)
    single_cls = hyperparameters.get('single_cls', False)
    
    if single_cls:
        new_groundTruth = []
        true, false = True, False
        for i, row in test_csv.iterrows():
            groundTruthRows = eval(row['imageAnnotations'])
            # pprint(groundTruthRows)

            for gTR in range(len(groundTruthRows)):
                for sOs in range(len(groundTruthRows[gTR]['selectedOptions'])):
                    groundTruthRows[gTR]['selectedOptions'][sOs]['value'] = 'obj' if groundTruthRows[gTR]['selectedOptions'][sOs]['value'] != 'root' else 'root'
            
            new_groundTruth.append(str(groundTruthRows))
        
        test_csv['imageAnnotations'] = new_groundTruth
    
    test_csv.to_csv('test_debug.csv')

    # Data structure to create the test_CSV for detection and segmentation models
    is_bb = True

    output_df = pd.DataFrame(
        columns=['resource_id', 'label', 'predicted', 'F1', 'Precision', 'Recall', 'Mean Average Precision',
                 'predictedAnnotations', 'groundTruthAnnotations'])

    filename_2_resource_id = {}
    for index, row in test_csv.iterrows():
        filename_2_resource_id[row['name']] = row['_id']

    # Data structure to calculate mAP and AUC
    segmentation_statistics = {}
    detections = []
    total_ground_truths = {}

    # Data structures to calculate model integrity
    # Contains the Total Count : Number of files that are passing the Confidence Score and IOU Threshold
    # Correct Count : Number of files that are passing the Confidence Score and IOU Threshold and the label is correct
    integrity_statistics = {}

    # Load the Yolo Model
    model = YOLO(models_path)

    # Load the JSON file generated during training and testing
    # This contains all the details
    with open(test_json_path) as f:
        ground_truths = json.load(f)

    # Create a dictionary mapping the category id to the category name
    imageid_2_name = {}
    for category in ground_truths["categories"]:
        imageid_2_name[category["id"]] = category["name"]
        integrity_statistics[category["name"]] = {"Total Count": 0, "Correct Count": 0}

    # Create a datastructure containing the image path, list of bounding boxes and list of labels
    database = {}
    for image in ground_truths["images"]:
        database[image["id"]] = {"id": image["id"], "file_name": image["file_name"], "annotations": [], "labels": []}

    for annotation in ground_truths["annotations"]:
        database[annotation["image_id"]]["annotations"].append(annotation["bbox"])
        database[annotation["image_id"]]["labels"].append(imageid_2_name[annotation["category_id"]])

    # A dictionary to store all the components of the confusion matrix
    class_precision = {}
    class_recall = {}
    class_f1 = {}
    class_tp = {}
    class_fp = {}
    class_fn = {}
    class_max_iou = {}
    class_average_iou = {}

    for c in imageid_2_name.values():
        class_tp[c] = 0
        class_fp[c] = 0
        class_fn[c] = 0
        class_max_iou[c] = []

    # Iterate over all the image in the database
    for image_id in database:

        # Data structures to calculate resource wise F1, Precision, Recall and mAP
        resource_wise_tp = 0
        resource_wise_fp = 0
        resource_wise_fn = 0
        resource_wise_ground_truths = {}
        resource_wise_detections = []

        file_path = os.path.join(dataset_path, database[image_id]["file_name"])
        ground_truth_annotations = database[image_id]["annotations"]
        ground_truth_labels = database[image_id]["labels"]
        
        if single_cls:
            for i in range(len(ground_truth_labels)):
                ground_truth_labels[i] = 'obj'

        # Run inference on the model and store the results
        results = model(file_path)
        predicted_boxes = []
        predicted_labels = []
        confidence_scores = []

        for result in results:
            predicted_boxes.extend(result.boxes.xyxy.tolist())
            confidence_scores.extend(result.boxes.conf.tolist())

            for label in result.boxes.cls.tolist():
                if single_cls:
                    predicted_labels.append('obj')
                else:
                    predicted_labels.append(model.names[label])

        # Non-Maximum Suppression
        try:
            predicted_boxes, predicted_labels, confidence_scores = nms(predicted_boxes, predicted_labels,
                                                                       confidence_scores)
        except:
            predicted_boxes, predicted_labels, confidence_scores = [], [], []
            

        # print("All predictions and ground truths")
        # print("predicted_boxes", predicted_boxes)
        # print("predicted_labels", predicted_labels)
        # print("confidence_scores", confidence_scores)
        # print("ground_truth_annotations", ground_truth_annotations)
        # print("ground_truth_labels", ground_truth_labels)

        # For each predicted box, find the maximum IoU with the ground truth boxes
        for pred_box, pred_class, conf_score in zip(predicted_boxes, predicted_labels, confidence_scores):

            # Finding the number of predictions that satisfies the Integrity Condition
            ious = [calculate_iou(pred_box, true_box) for true_box, true_class in
                    zip(ground_truth_annotations, ground_truth_labels)]
            max_iou = max(ious) if ious else 0
            if max_iou > Config.IOU_THRESHOLD and conf_score > hyperparameters['integrity_confidence_threshold']:
                integrity_statistics[pred_class]["Total Count"] += 1

            ious = [calculate_iou(pred_box, true_box) for true_box, true_class in
                    zip(ground_truth_annotations, ground_truth_labels) if
                    true_class == pred_class]
            max_iou = max(ious) if ious else 0

            # Adding the IOU to the class_iou dictionary
            class_max_iou[pred_class].append(max_iou)

            if max_iou > Config.IOU_THRESHOLD:  # Assuming IoU threshold for a true positive
                class_tp[pred_class] = class_tp[pred_class] + 1
                resource_wise_tp = resource_wise_tp + 1

                is_true_positive = True

                if conf_score > hyperparameters['integrity_confidence_threshold']:
                    integrity_statistics[pred_class]["Correct Count"] += 1

            else:
                class_fp[pred_class] = class_fp[pred_class] + 1
                resource_wise_fp = resource_wise_fp + 1

                is_true_positive = False

            # 'class_id', 'confidence', 'is_true_positive'
            detections.append({'class_id': pred_class, 'confidence': conf_score, 'is_true_positive': is_true_positive})
            resource_wise_detections.append({'class_id': pred_class, 'confidence': conf_score, 'is_true_positive': is_true_positive})

        # For each ground truth box, find the maximum IoU with the predicted boxes
        for true_box, true_class in zip(ground_truth_annotations, ground_truth_labels):

            total_ground_truths[true_class] = total_ground_truths.get(true_class, 0) + 1
            resource_wise_ground_truths[true_class] = resource_wise_ground_truths.get(true_class, 0) + 1

            ious = [calculate_iou(true_box, pred_box) for pred_box, pred_class in zip(predicted_boxes, predicted_labels)
                    if
                    pred_class == true_class]
            max_iou = max(ious) if ious else 0

            # Adding the IOU to the class_iou dictionary
            class_max_iou[true_class].append(max_iou)

            if max_iou <= Config.IOU_THRESHOLD:  # Assuming IoU threshold for a false negative
                class_fn[true_class] = class_fn[true_class] + 1
                resource_wise_fn = resource_wise_fn + 1

        resource_wise_precision = resource_wise_tp / max(resource_wise_tp + resource_wise_fp, 1)
        resource_wise_recall = resource_wise_tp / max(resource_wise_tp + resource_wise_fn, 1)
        resource_wise_f1_score = 2 * (resource_wise_precision * resource_wise_recall) / max(
            (resource_wise_precision + resource_wise_recall), 1e-8)
        resource_wise_aps = calculate_aps(resource_wise_ground_truths, resource_wise_detections)
        resource_wise_mAP = sum(resource_wise_aps.values()) / len(resource_wise_aps)

        # Add the results to the output CSV
        
        if single_cls:
            groundTruthRows = eval(
                test_csv[test_csv['name'] == database[image_id]["file_name"]]['imageAnnotations'].values[0]
            )
            
            for gTR in range(len(groundTruthRows)):
                for sOs in range(len(groundTruthRows[gTR]['selectedOptions'])):
                    groundTruthRows[gTR]['selectedOptions'][sOs]['value'] = 'obj' if groundTruthRows[gTR]['selectedOptions'][sOs]['value'] != 'root' else 'root'
            
            # pprint(groundTruthRows)
            
            groundTruthRows = json.dumps(groundTruthRows)
        else:
            groundTruthRows = test_csv[test_csv['name'] == database[image_id]["file_name"]][
                                                   'imageAnnotations'].values[0]
        
        try:
            output_df = output_df._append({'resource_id': filename_2_resource_id[database[image_id]["file_name"]],
                                           'label': 'obj',
                                           'predicted': 'obj' if round(resource_wise_mAP * 100, 2) == 100 else 'unknown',
                                           'groundTruthAnnotations': groundTruthRows,
                                           'predictedAnnotations': json.dumps(
                                               convert_bounding_boxes_to_image_annotations(predicted_boxes,
                                                                                           confidence_scores,
                                                                                           predicted_labels,
                                                                                           is_closed=True,
                                                                                           is_bb=is_bb)),
                                           'F1': round(resource_wise_f1_score * 100, 2),
                                           'Precision': round(resource_wise_precision * 100, 2),
                                           'Recall': round(resource_wise_recall * 100, 2),
                                           'Mean Average Precision': round(resource_wise_mAP * 100, 2)
                                           }, ignore_index=True)
        except:
            output_df = output_df._append({'resource_id': filename_2_resource_id[database[image_id]["file_name"]],
                                           'label': 'obj',
                                           'predicted': 'obj' if round(resource_wise_mAP * 100, 2) == 100 else 'unknown',
                                           'groundTruthAnnotations': groundTruthRows,
                                           'predictedAnnotations': json.dumps(
                                               convert_bounding_boxes_to_image_annotations(predicted_boxes,
                                                                                           confidence_scores,
                                                                                           predicted_labels,
                                                                                           is_closed=True,
                                                                                           is_bb=is_bb)),
                                           'F1': round(resource_wise_f1_score * 100, 2),
                                           'Precision': round(resource_wise_precision * 100, 2),
                                           'Recall': round(resource_wise_recall * 100, 2),
                                           'Mean Average Precision': round(resource_wise_mAP * 100, 2)
                                           }, ignore_index=True)

    # Calculate the precision, recall and f1 score, average IoU for each class
    for c in imageid_2_name.values():
        precision = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
        recall = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
        f1_score = 2 * (precision * recall) / max((precision + recall), 1e-8)

        class_average_iou[c] = sum(class_max_iou[c]) / len(class_max_iou[c])
        class_precision[c] = precision
        class_recall[c] = recall
        class_f1[c] = f1_score

    # Calculate the average precision, recall and f1 score
    average_precision = sum(class_precision.values()) / len(class_precision)
    average_recall = sum(class_recall.values()) / len(class_recall)
    average_f1_score = sum(class_f1.values()) / len(class_f1)

    final_results = {"Total": round(average_f1_score * 100, 2),
                     "Average Precision": round(average_precision * 100, 2),
                     "Average Recall": round(average_recall * 100, 2),
                     "Average F1 Score": round(average_f1_score * 100, 2),
                     "Average IoU": round(sum(class_average_iou.values()) / len(class_average_iou) * 100, 2),
                     }

    # Add the precision, recall and f1 score for each class to the final results
    for c in imageid_2_name.values():
        final_results[c + " Precision"] = round(class_precision[c] * 100, 2)
        final_results[c + " Recall"] = round(class_recall[c] * 100, 2)
        final_results[c + " F1 Score"] = round(class_f1[c] * 100, 2)

    # Adding details for the segmentation statistics
    # Calculate the mAP and AP for each class
    aps = calculate_aps(total_ground_truths, detections)
    mAP = sum(aps.values()) / len(aps)

    # Calculating the integrity score
    integrity_scores = {}
    integrity_total_count = 0
    integrity_correct_count = 0
    for c in integrity_statistics:
        integrity_total_count = integrity_total_count + integrity_statistics[c]["Total Count"]
        integrity_correct_count = integrity_correct_count + integrity_statistics[c]["Correct Count"]

        if integrity_statistics[c]["Total Count"] > 0:
            integrity_scores[c] = integrity_statistics[c]["Correct Count"] / integrity_statistics[c]["Total Count"]
        else:
            integrity_scores[c] = 0

    try:
        total_integrity = integrity_correct_count / integrity_total_count
    except ZeroDivisionError:
        total_integrity = 0

    # Overall statistics
    segmentation_statistics['F1'] = round(average_f1_score * 100, 2)
    segmentation_statistics['Precision'] = round(average_precision * 100, 2)
    segmentation_statistics['Recall'] = round(average_recall * 100, 2)
    segmentation_statistics['Mean Average Precision'] = round(mAP * 100, 2)
    segmentation_statistics['integrityFrequency'] = integrity_total_count
    segmentation_statistics['IntegrityAccuracy'] = round(total_integrity * 100, 2)
    segmentation_statistics['Average IoU'] = round(sum(class_average_iou.values()) / len(class_average_iou) * 100, 2)

    # Adding Class wise statistics
    segmentation_statistics['Labels'] = {}
    for c in imageid_2_name.values():
        segmentation_statistics['Labels'][c] = {}
        segmentation_statistics['Labels'][c]['F1'] = round(class_f1[c] * 100, 2)
        segmentation_statistics['Labels'][c]['Precision'] = round(class_precision[c] * 100, 2)
        segmentation_statistics['Labels'][c]['Recall'] = round(class_recall[c] * 100, 2)
        segmentation_statistics['Labels'][c]['Mean Average Precision'] = round(aps[c] * 100, 2)
        segmentation_statistics['Labels'][c]['integrityFrequency'] = integrity_statistics[c]["Total Count"]
        segmentation_statistics['Labels'][c]['IntegrityAccuracy'] = round(integrity_scores[c] * 100, 2)
        segmentation_statistics['Labels'][c]['Average IoU'] = round(class_average_iou[c] * 100, 2)

    # Storing the statistics in a file
    with open(statistics_file, 'w') as f:
        json.dump(segmentation_statistics, f)

    # Storing the results in a file
    print('this is output - ',output_file)
    output_df.to_csv(output_file, index=False)

    return final_results


if "__main__" == __name__:
    test_csv_path = "/home/anandhakrishnan/Projects/American-Ordinance/ao-fedramp/training-pods/Yolo-V58-Pod/datasets/test_654d3dee37c7830a7301a093/defaultDataSetCollection_654c6997cf33aed742036911_resources.csv"
    test_json_path = "/home/anandhakrishnan/Projects/American-Ordinance/ao-fedramp/training-pods/Yolo-V58-Pod/datasets/test_654d3dee37c7830a7301a093/654d3dee37c7830a7301a093.json"
    dataset_path = "/home/anandhakrishnan/Projects/American-Ordinance/ao-fedramp/training-pods/Yolo-V58-Pod/datasets/test_654d3dee37c7830a7301a093/images"
    models_path = "/home/anandhakrishnan/Projects/American-Ordinance/ao-fedramp/training-pods/Yolo-V58-Pod/runs/detect/train/weights/best.pt"
    output_file = "temp123.csv"
    statistics_file = "temp123.json"
    hyperparameters = {"project_type": "detection", "integrity_confidence_threshold": 0.5}
    main(test_csv_path, test_json_path, dataset_path, models_path, output_file, statistics_file, hyperparameters)
