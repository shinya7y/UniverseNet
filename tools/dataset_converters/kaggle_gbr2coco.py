# Modified from Adriano Passos's notebook
# https://www.kaggle.com/coldfir3/simple-yolox-dataset-generator-coco-json
import argparse
import json
import os
from ast import literal_eval

import pandas as pd


def convert_to_coco_dict(df, limit_wh=True):
    categories = [{'id': 1, 'name': 'cots', 'supercategory': 'animal'}]
    image_w = 1280
    image_h = 720

    annotation_id = 0
    images = []
    annotations = []
    for _, row in df.iterrows():
        images.append({
            'id': row['image_id_int'],
            'width': image_w,
            'height': image_h,
            'file_name': row['image_file_name'],
        })
        for bbox in row['annotations']:
            if limit_wh:
                orig_width = bbox['width']
                orig_height = bbox['height']
                bbox['width'] = min(bbox['width'], image_w - 1 - bbox['x'])
                bbox['height'] = min(bbox['height'], image_h - 1 - bbox['y'])
                if orig_width != bbox['width']:
                    print(f"Change box width {orig_width}->{bbox['width']}")
                if orig_height != bbox['height']:
                    print(f"Change box height {orig_height}->{bbox['height']}")
            annotations.append({
                'id': annotation_id,
                'image_id': row['image_id_int'],
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
    parser.add_argument('--train_json', default=None)
    parser.add_argument('--val_json', default=None)
    parser.add_argument('--val_video_id', default=2, type=int)
    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument('--json_indent', default=None, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.label_dir, exist_ok=True)
    prefix = 'instances_' + str(
        args.num_samples if args.num_samples else 'full')
    if args.train_json is None:
        args.train_json = f'{prefix}_not_video_{args.val_video_id}.json'
    if args.val_json is None:
        args.val_json = f'{prefix}_video_{args.val_video_id}.json'
    train_json_path = os.path.join(args.label_dir, args.train_json)
    val_json_path = os.path.join(args.label_dir, args.val_json)

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
        if num_empty_gt <= 1:
            raise NotImplementedError
        equally_spaced_indices = [
            int((num_img_without_anno - 1) * i / (num_empty_gt - 1))
            for i in range(num_empty_gt)
        ]
        without_anno = without_anno.iloc[equally_spaced_indices]
    df = pd.concat([with_anno, without_anno]).reset_index(drop=True)

    df['is_val_data'] = df['video_id'] == args.val_video_id
    df['annotations'] = df['annotations'].apply(literal_eval)
    df['image_id_int'] = df['video_id'] * 1000000 + df['video_frame']
    df['image_file_name'] = df.apply(
        lambda row: f"video_{row['video_id']}/{row['video_frame']}.jpg",
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
