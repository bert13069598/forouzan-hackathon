import argparse

import cv2
from glob import glob

import numpy as np
from tqdm import tqdm
import json

import os

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

    for label, bbox in class_id_bbox_info:
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Draw the bounding box
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Display the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_position = (x_min, y_min - 5)
        cv2.putText(image, label, text_position, font, font_scale, color, font_thickness)

    # Save the image with bounding boxes
    base_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_path, f"annotated_{base_name}")
    cv2.imwrite(output_image_path, image)

    
    

def convert(image_paths, output_path):
    for image_path in tqdm(image_paths):
        bboxes = find_bbox(image_path, output_path)




if __name__ == '__main__':
    path = '/home/elicer/dataset'
    image_paths = glob(path + f'/{args.label}/train/images/*')
    convert(image_paths, '/home/elicer/hackathon/gpu/crack/YOLO/check')
    # image_paths = glob(path + f'/{args.label}/val/images/*')
    # convert(image_paths, '/home/elicer/hackathon/gpu/crack/YOLO/check')

    print(set(cls_list))