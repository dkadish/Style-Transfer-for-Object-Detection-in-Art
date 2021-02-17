# Adapted from https://blog.roboflow.com/how-to-convert-annotations-from-voc-xml-to-coco-json/

import os
import argparse
import json
import shutil
import tarfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import re

from itertools import count
id = count()

def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split('\n')
    labels_ids = list(range(1, len(labels_str)+1))

    print('Found {} labels'.format(len(labels_str)))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True, is_peopleart=False):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    # img_id = os.path.splitext(img_name)[0]
    # if extract_num_from_imgid and isinstance(img_id, str):
    #     img_id = int(re.findall(r'\d+', img_id)[0])
    img_id = next(id)

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }

    if is_peopleart:
        folder = annotation_root.findtext('folder')
        image_info['style'] = folder
        image_info['file_name'] = '_'.join([folder, filename]) #TODO Check this!

    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True, is_peopleart=False, only_people_categories=False):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        try:
            # Read annotation xml
            ann_tree = ET.parse(a_path)
            ann_root = ann_tree.getroot()

            img_info = get_image_info(annotation_root=ann_root,
                                      extract_num_from_imgid=extract_num_from_imgid, is_peopleart=is_peopleart)
            img_id = img_info['id']
            output_json_dict['images'].append(img_info)

            for obj in ann_root.findall('object'):
                ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
                ann.update({'image_id': img_id, 'id': bnd_id})
                output_json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1
        except FileNotFoundError as e:
            print('No annotation for {}. Skipping.'.format(a_path))

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    if only_people_categories:
        output_json_dict['categories'] = [output_json_dict['categories'][0]]
        print('Only People Categories')
        print(output_json_dict['categories'])

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)

def save_and_compress_images(image_dir, image_out_dir, json_file):
    annotations = json.load(open(json_file))
    image_dir = Path(image_dir)
    image_out_dir = Path(image_out_dir)
    image_out_dir.mkdir(parents=True, exist_ok=True)
    for img in annotations['images']:
        shutil.copy(image_dir / img['file_name'], image_out_dir / img['file_name'])

    with tarfile.open('{}.tar.xz'.format(image_out_dir.parts[-1]), 'w:xz') as t:
        for img in image_out_dir.glob('*.jpg'):
            t.add(img, arcname=img.relative_to(image_out_dir.parent))


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_ids', type=str, default=None,
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_paths_list', type=str, default=None,
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='path to images directory.')
    parser.add_argument('--image_out_dir', type=str, default=None,
                        help='path to images output directory.')
    parser.add_argument('--labels', type=str, default=None,
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='output.json', help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    parser.add_argument('--is_peopleart', default=False, action='store_true',
                        help='Engage special extension for peopleart')
    parser.add_argument('--only_people_categories', default=False, action='store_true',
                        help='Engage special extension for peopleart')
    args = parser.parse_args()
    label2id = get_label2id(labels_path=args.labels)
    ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=False,
        is_peopleart=args.is_peopleart,
        only_people_categories=args.only_people_categories
    )
    save_and_compress_images(
        image_dir=args.image_dir,
        image_out_dir=args.image_out_dir,
        json_file=args.output
    )

#TODO this is a dump of some code. Not to be used as is.
from pycocotools.coco import COCO
from pathlib import Path
import json
def remove_nonperson_categories():
    path_to_annotations_style = Path('./annotations/style_val2017.json')

    # instantiate COCO specifying the annotations json path
    coco = COCO(path_to_annotations_style)
    coco.cats = coco.cats[1]
    coco.createIndex()

    with open('./annotations/person_val2017.json', 'w') as j:
        json.dump(coco.dataset, j)

if __name__ == '__main__':
    main()