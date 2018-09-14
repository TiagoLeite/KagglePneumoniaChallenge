# This is an attempt to implament the LB metric: mean avarage precision at different IoU threshold.
# It might be helpful for local validation. The `map_iou` function evaluates the metric on one image.
# 
# Idea borrowed from https://www.kaggle.com/raresbarbantan/f2-metric and is modified for this competition.
# 
# 
# I haven't thoroughly tested it so please comment if you found any bugs!

import numpy as np
import pandas as pd


# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds=None):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)

    # return None if both are empty, don't count the image in final evaluation (?)
    if thresholds is None:
        thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    # assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"

    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


res_pred = pd.read_csv('results.csv')
res_true = pd.read_csv('results_true.csv')

boxes_pred = res_pred['PredictionString']
joined_boxes_pred = list()
final_boxes_pred = list()

boxes_true = res_true['PredictionString']
joined_boxes_true = list()
final_boxes_true = list()

scores = list()

for k in range(len(boxes_pred)):

    if type(boxes_pred[k]) is str:
        new_boxes = [float(val) for val in boxes_pred[k].rstrip().split(' ')]
        joined_boxes_pred.append(new_boxes)
    else:
        joined_boxes_pred.append([])

for k in range(len(boxes_true)):
    if type(boxes_true[k]) is str:
        new_boxes = [float(val) for val in boxes_true[k].rstrip().split(' ')]
        joined_boxes_true.append(new_boxes)
    else:
        joined_boxes_true.append([])

for m in joined_boxes_pred:
    new = [m[k * 5 + 1:k * 5 + 5] for k in range(int(len(m) / 5))]
    final_boxes_pred.append(new)
    scores.append([m[k * 5] for k in range(int(len(m) / 5))])

for m in joined_boxes_true:
    new = [m[k * 4:k * 4 + 4] for k in range(int(len(m) / 4))]
    final_boxes_true.append(new)

sum = 0
count = 0

for k in range(len(scores)):

    print(final_boxes_true[k])
    print(final_boxes_pred[k])
    print(scores[k])

    boxes_true = np.array(final_boxes_true[k])
    boxes_pred = np.array(final_boxes_pred[k])
    score = scores[k]
    mapa = map_iou(boxes_true, boxes_pred, score)
    print(mapa, '\n')
    # print(type(mapa))

    if type(mapa) is float:
        sum += mapa
        count += 1

print("Final score:", round(sum/count, 3), "Count:", count)

