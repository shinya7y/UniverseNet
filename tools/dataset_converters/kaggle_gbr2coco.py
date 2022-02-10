# Modified from Adriano Passos's notebook
# https://www.kaggle.com/coldfir3/simple-yolox-dataset-generator-coco-json
import argparse
import json
import os
from ast import literal_eval

import pandas as pd


def convert_to_coco_dict(df):
    categories = [{'id': 1, 'name': 'cots', 'supercategory': 'animal'}]

    annotation_id = 0
    images = []
    annotations = []
    for image_index, row in df.iterrows():
        images.append({
            'id': image_index,
            'width': 1280,
            'height': 720,
            'file_name': f"{row['image_id']}.jpg",
        })
        for bbox in row['annotations']:
            annotations.append({
                'id': annotation_id,
                'image_id': image_index,
                'category_id': 1,
                'segmentation': None,
                'area': bbox['width'] * bbox['height'],
                'bbox': list(bbox.values()),
                'iscrowd': 0
            })
            annotation_id += 1

    coco_dict = {
        'categories': categories,
        'images': images,
        'annotations': annotations
    }
    return coco_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_csv',
        default='/kaggle/input/tensorflow-great-barrier-reef/train.csv')
    parser.add_argument(
        '--image_dir',
        default='/kaggle/input/tensorflow-great-barrier-reef/train_images')
    parser.add_argument(
        '--label_dir',
        default='/kaggle/data/tensorflow-great-barrier-reef/annotations')
    parser.add_argument('--label_train', default='instances_train2021.json')
    parser.add_argument('--label_val', default='instances_val2021.json')
    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument('--json_indent', default=None, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.label_dir, exist_ok=True)
    train_json_path = os.path.join(args.label_dir, args.label_train)
    val_json_path = os.path.join(args.label_dir, args.label_val)

    df = pd.read_csv(args.train_csv)
    with_anno = df[df['annotations'] != '[]']
    without_anno = df[df['annotations'] == '[]']
    num_img_with_anno = len(with_anno)
    num_img_without_anno = len(without_anno)
    print('num_img_with_anno:', num_img_with_anno)
    print('num_img_without_anno:', num_img_without_anno)

    if args.num_samples is not None:
        num_empty_gt = args.num_samples - num_img_with_anno
        print('undersampling images without annotation')
        print('num_samples', args.num_samples)
        print('num_empty_gt', num_empty_gt)
        if num_empty_gt < 0:
            raise NotImplementedError
        without_anno = without_anno.sample(num_empty_gt)
    df = pd.concat([with_anno, without_anno]).reset_index(drop=True)

    df['is_val_data'] = df['video_id'] == 2
    df['annotations'] = df['annotations'].apply(literal_eval)
    df['path'] = df.apply(
        lambda row:
        f"{args.image_dir}/video_{row['video_id']}/{row['video_frame']}.jpg",
        axis=1)

    print(df.tail())

    train_dict = convert_to_coco_dict(df[~df['is_val_data']])
    val_dict = convert_to_coco_dict(df[df['is_val_data']])

    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(
            train_dict,
            f,
            ensure_ascii=True,
            indent=args.json_indent,
            sort_keys=False)
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(
            val_dict,
            f,
            ensure_ascii=True,
            indent=args.json_indent,
            sort_keys=False)


if __name__ == '__main__':
    main()
