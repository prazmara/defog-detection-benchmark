from panopticapi.evaluation import pq_compute

import json
gt_json = json.load(open("/dataMeR2/mahdi/CV/foggy/panoptic.json"))
print(gt_json['categories'])

gt_json = "/dataMeR2/mahdi/CV/ground_truth/panoptic.json"
gt_folder = "/dataMeR2/mahdi/CV/ground_truth/"   # folder with GT panoptic PNGs
pred_json = "/dataMeR2/mahdi/CV/foggy/panoptic.json"
pred_folder = "/dataMeR2/mahdi/CV/foggy/"                   # folder with predicted PNGs

pq_res = pq_compute(gt_json, pred_json, gt_folder, pred_folder)
print("PQ results:", pq_res)      