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


def convert_to_yolo_format(image_width, image_height, bounding_boxes):
    yolo_boxes = []
    for label, bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2.0 / image_width
        y_center = (y_min + y_max) / 2.0 / image_height
        box_width = (x_max - x_min) / image_width
        box_height = (y_max - y_min) / image_height

        yolo_format = f"{label} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
        yolo_boxes.append(yolo_format)

    return yolo_boxes



cls_list=[]
def find_bbox(image_path, output_path):
    image = cv2.imread(image_path)
    h,w ,_=image.shape
    

    label_path = image_path.replace('images', 'anno').replace('jpg','json')

    with open(label_path, 'r') as file:
        json_data = json.load(file)
        annotations = json_data.get("Learning_Data_Info", {}).get("Annotations", [])
        class_id_bbox_info = [(anno.get("Class_ID"), anno.get("bbox", [])) for anno in annotations]

    # print(w, h)
    # print(class_id_bbox_info)

    bboxes = convert_to_yolo_format(w, h, class_id_bbox_info)

    # print(bboxes)


    with open(image_path.replace('images', 'labels').replace('jpg', 'txt'), 'w') as file:
        for bbox in bboxes:
            file.write(bbox)
    

def convert(image_paths, output_path):
    for image_path in tqdm(image_paths):
        find_bbox(image_path, output_path)




if __name__ == '__main__':
    path = '/home/elicer/dataset'
    image_paths = glob(path + f'/{args.label}/train/images/*')
    convert(image_paths, path + f'/{args.label}/train/labels')
    image_paths = glob(path + f'/{args.label}/val/images/*')
    convert(image_paths, path + f'/{args.label}/val/labels')

    print(set(cls_list))