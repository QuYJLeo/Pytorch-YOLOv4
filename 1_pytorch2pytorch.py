#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-19 17:20:11
#   Description : pytorch_yolov4
#
# ================================================================
import torch
from model.yolov4 import YOLOv4


def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights('yolov4.pt')
print('============================================================')


def copy1(idx, cccccccc):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'bn%d.weight' % idx
    keyword3 = 'bn%d.bias' % idx
    keyword4 = 'bn%d.running_mean' % idx
    keyword5 = 'bn%d.running_var' % idx
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            y = value
        elif keyword3 in key:
            b = value
        elif keyword4 in key:
            m = value
        elif keyword5 in key:
            v = value
    conv2, bn2 = cccccccc.conv, cccccccc.bn
    conv2.weight.data = torch.Tensor(w)
    bn2.weight.data = torch.Tensor(y)
    bn2.bias.data = torch.Tensor(b)
    bn2.running_mean.data = torch.Tensor(m)
    bn2.running_var.data = torch.Tensor(v)

def copy2(idx, cccccccc):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'conv%d.bias' % idx
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            b = value
    conv2 = cccccccc.conv
    conv2.weight.data = torch.Tensor(w)
    conv2.bias.data = torch.Tensor(b)


num_classes = 80
num_anchors = 3

yolo = YOLOv4(num_classes, num_anchors)


print('\nCopying...')
for i in range(1, 94, 1):
    try:
        copy1(i, yolo.get_layer('conv%.3d' % i))
    except:
        name = 'conv%.3d' % i
        print(name)
        continue
for i in range(95, 102, 1):
    copy1(i, yolo.get_layer('conv%.3d' % i))
for i in range(103, 110, 1):
    copy1(i, yolo.get_layer('conv%.3d' % i))

copy2(94, yolo.get_layer('conv094'))
copy2(102, yolo.get_layer('conv102'))
copy2(110, yolo.get_layer('conv110'))


k = 5
copy1(k, yolo.stackResidualBlock01.sequential.stack_1.conv1)
k += 1
copy1(k, yolo.stackResidualBlock01.sequential.stack_1.conv2)
k += 1


k = 12
copy1(k, yolo.stackResidualBlock02.sequential.stack_1.conv1)
k += 1
copy1(k, yolo.stackResidualBlock02.sequential.stack_1.conv2)
k += 1
copy1(k, yolo.stackResidualBlock02.sequential.stack_2.conv1)
k += 1
copy1(k, yolo.stackResidualBlock02.sequential.stack_2.conv2)
k += 1


k = 21
copy1(k, yolo.stackResidualBlock03.sequential.stack_1.conv1)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_1.conv2)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_2.conv1)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_2.conv2)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_3.conv1)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_3.conv2)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_4.conv1)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_4.conv2)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_5.conv1)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_5.conv2)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_6.conv1)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_6.conv2)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_7.conv1)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_7.conv2)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_8.conv1)
k += 1
copy1(k, yolo.stackResidualBlock03.sequential.stack_8.conv2)
k += 1



k = 42
copy1(k, yolo.stackResidualBlock04.sequential.stack_1.conv1)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_1.conv2)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_2.conv1)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_2.conv2)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_3.conv1)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_3.conv2)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_4.conv1)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_4.conv2)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_5.conv1)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_5.conv2)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_6.conv1)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_6.conv2)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_7.conv1)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_7.conv2)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_8.conv1)
k += 1
copy1(k, yolo.stackResidualBlock04.sequential.stack_8.conv2)
k += 1



k = 63
copy1(k, yolo.stackResidualBlock05.sequential.stack_1.conv1)
k += 1
copy1(k, yolo.stackResidualBlock05.sequential.stack_1.conv2)
k += 1
copy1(k, yolo.stackResidualBlock05.sequential.stack_2.conv1)
k += 1
copy1(k, yolo.stackResidualBlock05.sequential.stack_2.conv2)
k += 1
copy1(k, yolo.stackResidualBlock05.sequential.stack_3.conv1)
k += 1
copy1(k, yolo.stackResidualBlock05.sequential.stack_3.conv2)
k += 1
copy1(k, yolo.stackResidualBlock05.sequential.stack_4.conv1)
k += 1
copy1(k, yolo.stackResidualBlock05.sequential.stack_4.conv2)
k += 1


torch.save(yolo.state_dict(), 'pytorch_yolov4.pt')
print('\nDone.')


