import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Project imports
import core.datasets.metadata as metadata


def setup_all_datasets(dataset_dir):
    """
    Registers all datasets as instances from COCO

    Args:
        dataset_dir(str): path to dataset directory

    """
    setup_bdd_dataset(dataset_dir)
    setup_kitti_dataset(dataset_dir)
    setup_lyft_dataset(dataset_dir)


def setup_bdd_dataset(dataset_dir):
    """
    sets up BDD dataset following detectron2 coco instance format.
    """
    train_image_dir = os.path.join(dataset_dir, 'images', '100k', 'train')
    test_image_dir = os.path.join(dataset_dir, 'images', '100k', 'val')

    train_json_annotations = os.path.join(
        dataset_dir, 'labels', 'train_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'labels', 'val_coco_format.json')

    register_coco_instances(
        "bdd_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get("bdd_train").thing_classes = metadata.BDD_THING_CLASSES
    MetadataCatalog.get(
        "bdd_train").thing_dataset_id_to_contiguous_id = metadata.BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "bdd_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get("bdd_val").thing_classes = metadata.BDD_THING_CLASSES
    MetadataCatalog.get(
        "bdd_val").thing_dataset_id_to_contiguous_id = metadata.BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_kitti_dataset(dataset_dir):
    """
    sets up KITTI dataset following detectron2 coco instance format.
    """
    train_image_dir = os.path.join(
        dataset_dir, 'object', 'training', 'image_2')
    test_image_dir = os.path.join(dataset_dir, 'object', 'training', 'image_2')

    train_json_annotations = os.path.join(
        dataset_dir,
        'object',
        'training',
        'label2-COCO-Format',
        'train_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir,
        'object',
        'training',
        'label2-COCO-Format',
        'val_coco_format.json')

    register_coco_instances(
        "kitti_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "kitti_train").thing_classes = metadata.KITTI_THING_CLASSES
    MetadataCatalog.get(
        "kitti_train").thing_dataset_id_to_contiguous_id = metadata.KITTI_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "kitti_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "kitti_val").thing_classes = metadata.KITTI_THING_CLASSES
    MetadataCatalog.get(
        "kitti_val").thing_dataset_id_to_contiguous_id = metadata.KITTI_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_lyft_dataset(dataset_dir):
    """
    sets up KITTI dataset following detectron2 coco instance format.
    """
    test_image_dir = os.path.join(dataset_dir, 'train', 'image_2')

    test_json_annotations = os.path.join(
        dataset_dir,
        'train',
        'label2-COCO-Format',
        'val_coco_format.json')

    register_coco_instances(
        "lyft_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "lyft_val").thing_classes = metadata.BDD_THING_CLASSES
    MetadataCatalog.get(
        "lyft_val").thing_dataset_id_to_contiguous_id = metadata.BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID
