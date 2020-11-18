import argparse
import cv2
import json
import numpy as np
import os
import random

def create_coco_lists(
        ids_list,
        image_dir,
        annotations_dir,
        category_mapper,
        categories_to_use):
    """
    Creates lists in coco format to be written to JSON file.
    """
    images_list = []
    annotations_list = []
    count = 0

    for image_id in ids_list:

        image = cv2.imread(os.path.join(image_dir, image_id) + '.png')

        if image is None:
            continue

        images_list.append({'id': image_id,
                            'width': image.shape[1],
                            'height': image.shape[0],
                            'file_name': image_id + '.png',
                            'license': 1})

        gt_frame = np.loadtxt(
            os.path.join(
                annotations_dir,
                image_id) + '.txt',
            delimiter=' ',
            dtype=str,
            usecols=np.arange(start=0, step=1, stop=15))

        if gt_frame.shape[0] == 0:
            continue

        if len(gt_frame.shape) == 1:
            gt_frame = np.array([gt_frame.tolist()])

        class_filter = np.asarray(
            [inst.lower() in categories_to_use for inst in gt_frame[:, 0]])

        gt_frame = gt_frame[class_filter, :]

        category_names = gt_frame[:, 0]

        # Convert Dataset nouns to match bdd dataset
        category_names = ['car' if category_name ==
                          'car' else category_name for category_name in category_names]
        category_names = ['person' if category_name ==
                          'pedestrian' else category_name for category_name in category_names]
        category_names = ['bike' if category_name ==
                          'bicycle' else category_name for category_name in category_names]
        category_names = ['motor' if category_name ==
                          'motorcycle' else category_name for category_name in category_names]

        frame_boxes = np.array(gt_frame[:, 4:8]).astype(np.float)

        for bbox, category_name in zip(frame_boxes, category_names):
            bbox_coco = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]]

            annotations_list.append({'image_id': image_id,
                                     'id': count,
                                     'category_id': category_mapper[category_name],
                                     'bbox': bbox_coco,
                                     'area': bbox_coco[2] * bbox_coco[3],
                                     'iscrowd': 0})
            count += 1

    return images_list, annotations_list


def main(args):
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    dataset_dir = args.dataset_dir

    image_dir = os.path.expanduser(
        os.path.join(
            dataset_dir,
            'train',
            'image_2'))
    annotations_dir = os.path.expanduser(
        os.path.join(dataset_dir, 'train', 'label_2'))

    if args.output_dir is None:
        output_dir = os.path.expanduser(
            os.path.join(
                dataset_dir,
                'train',
                'label2-COCO-Format'))
    else:
        output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    licenses = [{'id': 1,
                 'name': 'none',
                 'url': 'none'}]

    # Uncomment if cyclist class is required
    categories_to_use = ('car', 'truck', 'bus', 'pedestrian', 'motorcycle', 'bicycle')

    categories = [{'id': 1, 'name': 'car', 'supercategory': 'vehicle'},
                  {'id': 2, 'name': 'bus', 'supercategory': 'vehicle'},
                  {'id': 3, 'name': 'truck', 'supercategory': 'vehicle'},
                  {'id': 4, 'name': 'person', 'supercategory': 'vehicle'},
                  {'id': 5, 'name': 'rider', 'supercategory': 'vehicle'},
                  {'id': 6, 'name': 'bike', 'supercategory': 'vehicle'},
                  {'id': 7, 'name': 'motor', 'supercategory': 'vehicle'}
                  ]

    category_mapper = {}
    category_keys = [category['name'] for category in categories]

    for category_name, category in zip(category_keys, categories):
        category_mapper[category_name] = category['id']

    # Process Training Labels
    train_ids_list = [f[:-4] for f in os.listdir(image_dir)]

    random.shuffle(train_ids_list)

    train_ids_list = train_ids_list[: 10000]

    training_image_list, training_annotation_list = create_coco_lists(
        train_ids_list, image_dir, annotations_dir, category_mapper, categories_to_use)

    json_dict_training = {'info': {'year': 2020},
                          'licenses': licenses,
                          'categories': categories,
                          'images': training_image_list,
                          'annotations': training_annotation_list}

    training_file_name = os.path.join(output_dir, 'train_coco_format.json')

    with open(training_file_name, 'w') as outfile:
        json.dump(json_dict_training, outfile)

    print("Finished processing Lyft training data!")


if __name__ == "__main__":
    # Create arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=str,
        help='bdd100k dataset directory')

    parser.add_argument(
        "--output-dir",
        required=False,
        type=str,
        help='converted dataset write directory')

    args = parser.parse_args()
    main(args)
