import json
import torch

def action2verb(action):
    action2verb_dict = {
        0: 0,
        1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
        9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,
        17: 3, 18: 3, 19: 3,
        20: 4, 21: 4, 22: 4,
        23: 5,
        24: 6, 25: 6, 26: 6, 27: 6,
        28: 7, 29: 7, 30: 7,
        31: 8, 32: 8,
        33: 9, 34: 9,
        35: 10,
        36: 11,
    }
    return action2verb_dict[action]

def read_action_test1(file_path):
    data = {}
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            id = int(parts[0])
            action_label = int(parts[2])
            data[id] = action_label
    return data

def read_json_labels(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return {int(k): int(v) for k, v in data.items() if k.isdigit()}

def read_obj_gt(file_path):
    return torch.load(file_path).tolist()

def calculate_accuracy(gt_dict, pred_dict, name="Object"):
    correct = 0
    total = 0
    errors = []

    for id, gt_label in gt_dict.items():
        if id in pred_dict:
            pred_label = pred_dict[id]
            if pred_label == gt_label:
                correct += 1
            else:
                errors.append((id, pred_label, gt_label))
            total += 1

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"{name} Accuracy: {accuracy:.2f}%")

    if errors:
        print(f"\n{name} Prediction Errors ({len(errors)} cases):")
        for id, pred, gt in errors:
            print(f"  ID {id}: predicted = {pred}, ground truth = {gt}")

    return accuracy

if __name__ == '__main__':
    action_gt_file = 'D:/Downloads/dataset/h2o/label_split/action_test1.txt'
    obj_gt_file = '../data/h2o/h2o_pth/test/obj_labels.pth'

    action_pred_file = '../exp/h2o/action_labels.json'
    verb_pred_file = '../exp/h2o/verb_labels.json'
    obj_pred_file = '../exp/h2o/obj_labels.json'

    action_gt = read_action_test1(action_gt_file)
    verb_gt = {id: action2verb(a) for id, a in action_gt.items()}
    obj_gt = read_obj_gt(obj_gt_file)

    action_pred = read_json_labels(action_pred_file)
    verb_pred = read_json_labels(verb_pred_file)
    obj_pred = read_json_labels(obj_pred_file)

    obj_gt_dict = {i + 1: label for i, label in enumerate(obj_gt)}

    action_acc = calculate_accuracy(action_gt, action_pred, name="Action")
    verb_acc = calculate_accuracy(verb_gt, verb_pred, name="Verb")
    obj_acc = calculate_accuracy(obj_gt_dict, obj_pred, name="Object")
