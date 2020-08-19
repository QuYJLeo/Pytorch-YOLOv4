#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-19 17:20:11
#   Description : pytorch_yolov4
#
# ================================================================
import copy
import torch
from tools.cocotools import get_classes, catid2clsid, clsid2catid
from model.yolov4 import YOLOv4
from model.decode_np import Decode
import json
from tools.cocotools import eval

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # classes_path = 'data/voc_classes.txt'
    classes_path = 'data/coco_classes.txt'
    # model_path可以是'pytorch_yolov4.pt'、'./weights/step00001000.pt'这些。
    model_path = 'pytorch_yolov4.pt'
    # model_path = './weights/step00001000.pt'

    # input_shape越大，精度会上升，但速度会下降。
    # input_shape = (320, 320)
    # input_shape = (416, 416)
    input_shape = (608, 608)
    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.001
    nms_thresh = 0.45
    # 是否画出验证集图片
    draw_image = False
    # 验证时的批大小
    eval_batch_size = 2

    # 验证集图片的相对路径
    # eval_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'
    # anno_file = 'annotation_json/voc2012_val.json'
    eval_pre_path = '../COCO/val2017/'
    anno_file = '../COCO/annotations/instances_val2017.json'
    with open(anno_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            images = dataset['images']

    num_anchors = 3
    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)

    yolo = YOLOv4(num_classes)
    if torch.cuda.is_available():  # 如果有gpu可用，模型（包括了权重weight）存放在gpu显存里
        yolo = yolo.cuda()
    yolo.load_state_dict(torch.load(model_path))
    yolo.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.

    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _clsid2catid = {}
        for k in range(num_classes):
            _clsid2catid[k] = k

    _decode = Decode(conf_thresh, nms_thresh, input_shape, yolo, all_classes)
    box_ap = eval(_decode, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image)

