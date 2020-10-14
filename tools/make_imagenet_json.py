'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 17:27:16
@FilePath       : /ImageCls.detectron2/tools/make_imagenet_json.py
@Description    : 
'''

import re
import os
import os.path as osp
import argparse
import json
from tqdm import tqdm
import numpy as np
import cv2

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

ARCHIVE_META = {
    'train': 'train_set',
    'val': 'test_set',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Make imagenet dataset d2-style")
    parser.add_argument('--root', type=str, help="ImageNet root directory")
    parser.add_argument('--save', type=str, help="Result saving directory")

    args = parser.parse_args()
    if not osp.exists(args.save):
        os.makedirs(args.save)

    assert osp.exists(osp.join(args.root, ARCHIVE_META['train']))
    assert osp.exists(osp.join(args.root, ARCHIVE_META['val']))

    return args


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def accumulate_imagenet_json(image_root, phase):
    # accumulate infos
    classes = [i for i in range(0, 16)]
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    json_file = osp.join(image_root, 'labels_' + phase + '.txt')
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(tqdm(imgs_anns.keys())):
        # FIXME: fake data has images_test\\ or images_train\\ appended in front of keys
        key = re.sub("images_\w{4,5}\\\\", "", v)
        filename = osp.join(image_root, phase + '_set', key)
        # height, width = cv2.imread(filename).shape[:2]

        record = {
            "file_name": osp.abspath(filename),  # Using abs path, ignore image root, less flexibility
            "image_id": idx,  # fake data only has a max of 1 transformed grid segment
            "label": imgs_anns[v]["index"],
        }
        dataset_dicts.append(record)

    return dataset_dicts, class_to_idx


def main(args):
    # TODO: use GroupKFold https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
    # to split the train/test/val datasets
    # Accumulate train
    dataset_dicts_train, class_to_idx = accumulate_imagenet_json(args.root, phase='train')
    # Accumulate val
    dataset_dicts_val, _ = accumulate_imagenet_json(args.root, phase='test')
    # Save
    with open(osp.join(args.save, "imagenet_detectron2_train.json"), "w") as w_obj:
        json.dump(dataset_dicts_train, w_obj)
    with open(osp.join(args.save, "imagenet_detectron2_val.json"), "w") as w_obj:
        json.dump(dataset_dicts_val, w_obj)


if __name__ == "__main__":
    args = parse_args()
    main(args)
