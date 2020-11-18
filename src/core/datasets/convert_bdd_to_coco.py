import argparse
import json
import os

from collections import defaultdict

# BDD dataset has fixed image width and height.
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def group_by_key(detections, key):
    """
    Groups detections based on a key. Can be used to group detections by image id, category, etc.

    Args:
        detections: bdd JSON detection format
        key (str): key for detections to be grouped in.

    Returns:
        groups(Dict): a dictionary where detections are grouped by key.
    """
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups


def create_coco_lists(input_labels,
                      category_keys,
                      category_mapper):
    """
    Creates lists in coco format to be written to JSON file.
    """
    grouped_per_frame = group_by_key(input_labels, 'name')

    images_list = []
    annotations_list = []

    count = 0

    for im_id, frame in enumerate(grouped_per_frame):
        images_list.append({'id': im_id,
                            'width': IMAGE_WIDTH,
                            'height': IMAGE_HEIGHT,
                            'file_name': frame,
                            'license': 1})

        annotations = grouped_per_frame[frame]

        for annotation in annotations:
            if annotation['category'] in category_keys:
                bbox = annotation['bbox']
                bbox_coco = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1]]
                annotations_list.append({'image_id': im_id,
                                         'id': count,
                                         'category_id': category_mapper[annotation['category']],
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
    train_label_file_name = os.path.join(
        dataset_dir, 'labels', 'train') + '.json'
    val_label_file_name = os.path.join(
        dataset_dir, 'labels', 'val') + '.json'

    if args.output_dir is None:
        output_dir = os.path.expanduser(os.path.join(dataset_dir, 'labels'))
    else:
        output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    licenses = [{'id': 1,
                 'name': 'none',
                 'url': 'none'}]

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
    training_labels = json.load(
        open(os.path.expanduser(train_label_file_name), 'r'))
    training_image_list, training_annotation_list = create_coco_lists(
        training_labels, category_keys, category_mapper)

    json_dict_training = {'info': {'year': 2020},
                          'licenses': licenses,
                          'categories': categories,
                          'images': training_image_list,
                          'annotations': training_annotation_list}

    training_file_name = os.path.join(output_dir, 'train_coco_format.json')
    with open(training_file_name, 'w') as outfile:
        json.dump(json_dict_training, outfile)

    print("Finished processing BDD training data!")

    # Process Validation Labels
    validation_labels = json.load(
        open(os.path.expanduser(val_label_file_name), 'r'))
    validation_image_list, validation_annotation_list = create_coco_lists(
        validation_labels, category_keys, category_mapper)

    json_dict_validation = {'info': {'year': 2020},
                            'licenses': licenses,
                            'categories': categories,
                            'images': validation_image_list,
                            'annotations': validation_annotation_list}

    validation_file_name = os.path.join(output_dir, 'val_coco_format.json')
    with open(validation_file_name, 'w') as outfile:
        json.dump(json_dict_validation, outfile)

    print("Converted BDD to COCO format!")


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
