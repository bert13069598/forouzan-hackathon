import argparse

import cv2
from glob import glob

import numpy as np
from tqdm import tqdm
import json

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', type=str, required=True)
    return parser.parse_args()


args = argparser()


def xyxy2yolo(bbox, shape):
    height, width, _ = shape
    x_min, y_min, x_max, y_max = bbox

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    
    cx /= width
    cy /= height
    w /= width
    h /= height
    return [cx, cy, w, h]
cls_list=[]
def find_bbox(image_path, output_path):
    image = cv2.imread(image_path)
    w,h,_=image.shape
    

    label_path = image_path.replace('images', 'anno').replace('jpg','json')

    bboxes = []
    with open(label_path, 'r') as file:
        json_data = json.load(file)
        annotations = json_data.get("Learning_Data_Info", {}).get("Annotations", [])
        class_id_bbox_info = [(anno.get("Class_ID"), anno.get("bbox", [])) for anno in annotations]

    with open(image_path.replace('images', 'labels').replace('jpg', 'txt'), 'w') as file:
        for cls, bbox in class_id_bbox_info:
            cls_list.append(cls)
            bbox=xyxy2yolo(bbox, image.shape)
            file.write(f"{cls} {' '.join(map(lambda x: format(x, '.6f'), bbox))}")
    

def convert(image_paths, output_path):
    for image_path in tqdm(image_paths):
        bboxes = find_bbox(image_path, output_path)




if __name__ == '__main__':
    path = '/home/elicer/dataset'
    image_paths = glob(path + f'/{args.label}/train/images/*')
    convert(image_paths, path + f'/{args.label}/train/labels')
    image_paths = glob(path + f'/{args.label}/val/images/*')
    convert(image_paths, path + f'/{args.label}/val/labels')

    print(set(cls_list))