import argparse
import cv2
import json
import numpy as np
import os


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

        if len(gt_frame.shape) == 1:
            gt_frame = np.array([gt_frame.tolist()])

        class_filter = np.asarray(
            [inst.lower() in categories_to_use for inst in gt_frame[:, 0]])

        gt_frame = gt_frame[class_filter, :]

        category_names = gt_frame[:, 0]

        # Convert Dataset nouns to match bdd dataset
        category_names = ['car' if category_name ==
                          'Car' else category_name for category_name in category_names]
        category_names = ['person' if category_name ==
                          'Pedestrian' else category_name for category_name in category_names]

        # Uncomment if cyclist class is required
        # category_names = ['bike' if category_name ==
        #                   'Cyclist' else category_name for category_name in category_names]

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
            'object',
            'training',
            'image_2'))
    annotations_dir = os.path.expanduser(
        os.path.join(dataset_dir, 'object', 'training', 'label_2'))

    train_ids_file = os.path.expanduser(
        os.path.join(
            dataset_dir,
            'object',
            'train') + '.txt')
    val_ids_file = os.path.expanduser(
        os.path.join(
            dataset_dir,
            'object',
            'val') + '.txt')

    if args.output_dir is None:
        output_dir = os.path.expanduser(
            os.path.join(
                dataset_dir,
                'object',
                'training',
                'label2-COCO-Format'))
    else:
        output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    licenses = [{'id': 1,
                 'name': 'none',
                 'url': 'none'}]

    # Uncomment if cyclist class is required
    categories_to_use = ('car', 'pedestrian')#, 'cyclist')

    categories = [{'id': 1, 'name': 'car', 'supercategory': 'vehicle'},
                  {'id': 2, 'name': 'person', 'supercategory': 'person'}]#,
                  #{'id': 3, 'name': 'bike', 'supercategory': 'vehicle'}]

    category_mapper = {}
    category_keys = [category['name'] for category in categories]

    for category_name, category in zip(category_keys, categories):
        category_mapper[category_name] = category['id']

    # Process Training Labels
    with open(train_ids_file, 'r') as f:
        train_ids_list = [line for line in f.read().splitlines()]

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

    print("Finished processing PascalVOC training data!")

    # Process Validation Labels
    with open(val_ids_file, 'r') as f:
        val_ids_list = [line for line in f.read().splitlines()]

    validation_image_list, validation_annotation_list = create_coco_lists(
        val_ids_list, image_dir, annotations_dir, category_mapper, categories_to_use)

    json_dict_validation = {'info': {'year': 2020},
                            'licenses': licenses,
                            'categories': categories,
                            'images': validation_image_list,
                            'annotations': validation_annotation_list}

    validation_file_name = os.path.join(output_dir, 'val_coco_format.json')
    with open(validation_file_name, 'w') as outfile:
        json.dump(json_dict_validation, outfile)

    print("Converted PascalVOC to COCO format!")


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
