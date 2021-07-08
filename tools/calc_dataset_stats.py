import argparse
import json
import os
import pprint

import pandas as pd


def calc_coco_format_stats(ann_file,
                           verbose=False,
                           num_print=0,
                           ignore_iscrowd=False):
    print('annotation file:', ann_file)
    with open(ann_file) as f:
        anno = json.load(f)

    print('keys:', anno.keys())
    if verbose and ('categories' in anno):
        print('categories:')
        pprint.pprint(anno['categories'])

    image_id_to_area = {}
    if 'images' in anno:
        num_images = len(anno['images'])
        print('num_images:', num_images)
        if num_print > 0:
            pprint.pprint(anno['images'][:num_print])
        for annim in anno['images']:
            image_id_to_area[annim['id']] = annim['height'] * annim['width']

    if 'annotations' not in anno:
        print('annotations not found')
        return

    bboxes = anno['annotations']
    if ignore_iscrowd:
        bboxes = [box for box in bboxes if not box.get('iscrowd', False)]
    num_bboxes = len(bboxes)
    print('num_bboxes:', num_bboxes)
    df = pd.DataFrame(bboxes)

    if 'categories' in anno:
        category_id_to_name = {
            category['id']: category['name']
            for category in anno['categories']
        }
        df['category_name'] = df['category_id'].map(category_id_to_name)
        if verbose:
            num_bboxes_per_category = df['category_name'].value_counts()
            with pd.option_context('display.max_rows', None):
                print('num_bboxes_per_category:')
                print(num_bboxes_per_category)

        df['bbox_area'] = df['bbox'].str[2] * df['bbox'].str[3]
        # print(df[df['bbox_area'] < 1])
        if image_id_to_area:
            df['image_area'] = df['image_id'].map(image_id_to_area)
            df['relative_bbox_area'] = df['bbox_area'] / df['image_area']

    if num_print > 0:
        print(df.head(num_print))

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'ann_files', nargs='+', help='COCO-format annotation file')
    parser.add_argument(
        '--save_pkl',
        action='store_true',
        help='whether to save bbox dataframe as pkl')
    parser.add_argument(
        '--verbose', action='store_true', help='whether to print details')
    parser.add_argument(
        '--num_print',
        type=int,
        default=0,
        help='number to print the first few examples')
    args = parser.parse_args()

    ann_basenames = []
    dfs = []
    for ann_file in args.ann_files:
        ann_basename = os.path.splitext(os.path.basename(ann_file))[0]
        ann_basenames.append(ann_basename)
        df = calc_coco_format_stats(
            ann_file, verbose=args.verbose, num_print=args.num_print)
        dfs.append(df)
    df_concat = pd.concat(dfs)
    percentiles = [.01, .05, .1, .25, .5, .75, .9, .95, .99]
    print(df_concat.describe(percentiles=percentiles))

    pkl_dirname = os.path.dirname(args.ann_files[0])
    pkl_basename = '_'.join(ann_basenames) + '_stats.pkl'
    pkl_file = os.path.join(pkl_dirname, pkl_basename)
    if args.save_pkl:
        print('output pkl file:', pkl_file)
        df_concat.to_pickle(pkl_file)


if __name__ == '__main__':
    main()
