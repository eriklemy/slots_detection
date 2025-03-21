import os
import cv2

import yaml
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path

import albumentations as A
from ultralytics import YOLO

import matplotlib.pyplot as plt

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict

import torch

device = 0 if torch.cuda.is_available() else 'cpu'

def load_yaml(yaml_file):
    """Load YAML file and return contents."""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def plot_image_results(model_path):
    """
    Load and plot the image results from the specified model directory.
    """
    results_path = Path(f"{model_path}/")
    results = cv2.imread(str(results_path / "results.png"))
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(results, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def plot_training_results(results_path):
    """
    Load and plot the training results from the specified CSV file.
    """
    results = pd.read_csv(results_path)
    print(results.head())
    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(results['train/box_loss'], label='train/box_loss')
    plt.plot(results['val/box_loss'], label='val/box_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Box Loss')
    plt.legend()

    # mAP plot
    plt.subplot(1, 2, 2)
    plt.plot(results['metrics/mAP50(B)'], label='metrics/mAP50(B)')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_color_histogram(color_means, title):
    """ Plot histogram of color channel means. """
    plt.figure(figsize=(10, 5))
    plt.hist(color_means[:, 0], bins=50, alpha=0.5, color='b', label='Blue')
    plt.hist(color_means[:, 1], bins=50, alpha=0.5, color='g', label='Green')
    plt.hist(color_means[:, 2], bins=50, alpha=0.5, color='r', label='Red')
    plt.title(title)
    plt.xlabel('Pixel Intensity Mean')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_color_channel_distribution(color_means, title):
    """ Plot boxplot of color channel distributions. """
    plt.figure(figsize=(10, 5))
    plt.boxplot([color_means[:, 0], color_means[:, 1], color_means[:, 2]])
    plt.title('Color Channel Distributions (BGR)')
    plt.xticks([1, 2, 3], ['Blue', 'Green', 'Red'])
    plt.ylabel('Pixel Intensity Mean')
    plt.show()

def create_sahi_dataset(data_yaml_path, output_dir, slice_size=640, overlap_ratio=0.2):
    """
    Create a SAHI dataset from the original dataset.

    Args:
        data_yaml_path: Path to the data.yaml file
        output_dir: Directory to save the augmented dataset
        slice_size: Size of the slices
        overlap_ratio: Overlap ratio between slices
    """
    data_config = load_yaml(data_yaml_path)

    # Create output directories
    sahi_train_dir = os.path.join(output_dir, "train")
    sahi_val_dir = os.path.join(output_dir, "val")
    sahi_train_images_dir = os.path.join(sahi_train_dir, "images")
    sahi_train_labels_dir = os.path.join(sahi_train_dir, "labels")
    sahi_val_images_dir = os.path.join(sahi_val_dir, "images")
    sahi_val_labels_dir = os.path.join(sahi_val_dir, "labels")

    os.makedirs(sahi_train_images_dir, exist_ok=True)
    os.makedirs(sahi_train_labels_dir, exist_ok=True)
    os.makedirs(sahi_val_images_dir, exist_ok=True)
    os.makedirs(sahi_val_labels_dir, exist_ok=True)

    # Determine dataset structure
    dataset_dir = os.path.dirname(data_yaml_path)
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    # Process training and validation images
    process_dataset_split(train_dir, sahi_train_images_dir, sahi_train_labels_dir, slice_size, overlap_ratio)
    process_dataset_split(val_dir, sahi_val_images_dir, sahi_val_labels_dir, slice_size, overlap_ratio)

    # Create a new data.yaml
    new_data_yaml = os.path.join(output_dir, "data.yaml")
    data_config['path'] = output_dir
    data_config['train'] = os.path.join(output_dir, "train", "images")
    data_config['val'] = os.path.join(output_dir, "val", "images")

    with open(new_data_yaml, 'w') as f:
        yaml.dump(data_config, f)

    print(f"SAHI-augmented dataset created at {output_dir}")
    print(f"New data.yaml file: {new_data_yaml}")

    return new_data_yaml

def process_dataset_split(source_dir, output_images_dir, output_labels_dir, slice_size, overlap_ratio):
    """
    Process a dataset split (train or val) with SAHI slicing.

    Args:
        source_dir: Source directory containing images and labels
        output_images_dir: Output directory for images
        output_labels_dir: Output directory for labels
        slice_size: Size of the slices
        overlap_ratio: Overlap ratio between slices
    """
    source_images_dir = os.path.join(source_dir, "images")
    source_labels_dir = os.path.join(source_dir, "labels")

    # If directories don't exist, assume images and labels are in the same directory
    if not os.path.exists(source_images_dir):
        source_images_dir = source_dir
        source_labels_dir = source_dir.replace("images", "labels")

    for img_file in tqdm(os.listdir(source_images_dir), desc=f"Processing {os.path.basename(source_dir)}"):
        if not (img_file.endswith('.jpg') or img_file.endswith('.png') or img_file.endswith('.jpeg')):
            continue

        img_path = os.path.join(source_images_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(source_labels_dir, label_file)

        shutil.copy(img_path, os.path.join(output_images_dir, img_file))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_labels_dir, label_file))

        # Perform SAHI slicing
        try:
            image = cv2.imread(img_path)
            h, w = image.shape[:2]

            if h > slice_size or w > slice_size:  # Only slice if image is larger than slice size
                slice_and_save(img_path, label_path, output_images_dir, output_labels_dir,
                              slice_size, overlap_ratio, base_name)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

def slice_and_save(img_path, label_path, output_images_dir, output_labels_dir, slice_size, overlap_ratio, base_name):
    """
    Slice an image and its annotations using SAHI-style slicing and save to output directories.

    Args:
        img_path: Path to the input image
        label_path: Path to the input label (YOLO format)
        output_images_dir: Output directory for sliced images
        output_labels_dir: Output directory for sliced labels
        slice_size: Size of the slices
        overlap_ratio: Overlap ratio between slices
        base_name: Base name for the output files
    """
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Get YOLO labels -> [class_id, x_center, y_center, width, height]
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                width = float(parts[3]) * w
                height = float(parts[4]) * h

                # Convert to xmin, ymin, xmax, ymax
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2

                annotations.append({
                    'class_id': class_id,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })

    # Calculate the number of slices
    stride = int(slice_size * (1 - overlap_ratio))
    num_x_slices = max(1, int(np.ceil((w - slice_size) / stride)) + 1)
    num_y_slices = max(1, int(np.ceil((h - slice_size) / stride)) + 1)

    slice_idx = 0

    # Create slices
    for y_idx in range(num_y_slices):
        for x_idx in range(num_x_slices):
            # Calculate coordinates for this slice
            x_start = min(x_idx * stride, w - slice_size)
            y_start = min(y_idx * stride, h - slice_size)
            x_end = x_start + slice_size
            y_end = y_start + slice_size

            # Extract slice
            slice_img = image[y_start:y_end, x_start:x_end]
            slice_h, slice_w = slice_img.shape[:2]

            # Create labels for the slice
            slice_annotations = []
            for anno in annotations:
                # Check if annotation intersects with the slice
                if (anno['xmin'] < x_end and anno['xmax'] > x_start and
                    anno['ymin'] < y_end and anno['ymax'] > y_start):

                    # Clip the bounding box to the slice
                    xmin_clipped = max(0, anno['xmin'] - x_start)
                    ymin_clipped = max(0, anno['ymin'] - y_start)
                    xmax_clipped = min(slice_w, anno['xmax'] - x_start)
                    ymax_clipped = min(slice_h, anno['ymax'] - y_start)

                    width_clipped = xmax_clipped - xmin_clipped
                    height_clipped = ymax_clipped - ymin_clipped

                    # Check if the area is significant
                    original_area = (anno['xmax'] - anno['xmin']) * (anno['ymax'] - anno['ymin'])
                    clipped_area = width_clipped * height_clipped

                    # Only include if at least 30% of original bbox is visible
                    if clipped_area >= 0.3 * original_area:
                        # Convert back to YOLO format (class_id, x_center, y_center, width, height)
                        x_center = (xmin_clipped + xmax_clipped) / (2 * slice_w)
                        y_center = (ymin_clipped + ymax_clipped) / (2 * slice_h)
                        width = (xmax_clipped - xmin_clipped) / slice_w
                        height = (ymax_clipped - ymin_clipped) / slice_h

                        slice_annotations.append(f"{anno['class_id']} {x_center} {y_center} {width} {height}")

            if slice_annotations:
                slice_file = f"{base_name}_slice_{slice_idx}.jpg"
                cv2.imwrite(os.path.join(output_images_dir, slice_file), slice_img)

                slice_label_file = f"{base_name}_slice_{slice_idx}.txt"
                with open(os.path.join(output_labels_dir, slice_label_file), 'w') as f:
                    f.write('\n'.join(slice_annotations))

                slice_idx += 1

def apply_albumentations_augmentations(data_yaml_path):
    """
    Apply albumentations augmentations to the dataset.

    Args:
        data_yaml_path: Path to the data.yaml file
    """
    data_config = load_yaml(data_yaml_path)
    train_dir = data_config['train']

    # If train_dir is a relative path, make it absolute based on the yaml location
    if not os.path.isabs(train_dir):
        base_dir = os.path.dirname(data_yaml_path)
        train_dir = os.path.join(base_dir, train_dir)

    images_dir = train_dir
    if os.path.isdir(os.path.join(train_dir, "images")):
        images_dir = os.path.join(train_dir, "images")

    labels_dir = images_dir.replace("images", "labels")

    # Augmentation pipeline
    aug_transforms = A.Compose([
        A.RandomRotate90(p=0.5),                       # random 90 degree rotation with 50% probability to happen
        A.HorizontalFlip(p=0.5),                       # horizontal flip with 50% probability
        A.RandomBrightnessContrast(p=0.3),             # brightness/contrast with 30% probability
        A.HueSaturationValue(p=0.2),                   # hue/saturation/value with 20% probability
        A.RandomGamma(gamma_limit=(60, 115), p=0.3),   # random gamma correction with 30% probability and medium gamma range
        A.GaussNoise(p=0.3),                           # gaussian noise with 30% probability
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in tqdm(images, desc="Applying augmentations"):
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        # Read labels
        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)

        if not bboxes:
            continue

        # Apply augmentations 2 times (to create 3x more data and allow for more variety - since some augmentations are random)
        for i in range(2):
            try:
                augmented = aug_transforms(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_labels = augmented['class_labels']

                # Save augmented image and labels
                aug_img_file = f"{base_name}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(images_dir, aug_img_file), aug_image)

                aug_label_file = f"{base_name}_aug_{i}.txt"
                with open(os.path.join(labels_dir, aug_label_file), 'w') as f:
                    for j, bbox in enumerate(aug_bboxes):
                        f.write(f"{int(aug_class_labels[j])} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            except Exception as e:
                print(f"Error augmenting {img_file}: {e}")
                continue


def evaluate_with_sahi(model_path, data_yaml_path, slice_size=640, overlap_ratio=0.2):
    """
    Evaluate the model with SAHI.

    Args:
        model_path: Path to the trained model weights
        data_yaml_path: Path to the data.yaml file
        slice_size: Size of the slices
        overlap_ratio: Overlap ratio between slices
    """
    data_config = load_yaml(data_yaml_path)
    val_dir = data_config['val']

    if not os.path.isabs(val_dir):
        base_dir = os.path.dirname(data_yaml_path)
        val_dir = os.path.join(base_dir, val_dir)

    # Load the model using SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=0.3,
        device=device
    )

    # Get validation images
    if os.path.isdir(os.path.join(val_dir, "images")):
        val_dir = os.path.join(val_dir, "images")

    images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Evaluate with SAHI
    for img_path in tqdm(images, desc="Evaluating with SAHI"):
        image = read_image(img_path)
        img_name = os.path.basename(img_path).split('.')[0]

        standard_prediction = get_prediction(
            image=image,
            detection_model=detection_model,
        )
        standard_prediction.export_visuals(export_dir="out/standard/", file_name=img_name, rect_th=2)

        sliced_prediction = get_sliced_prediction(
            image=image,
            detection_model=detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio
        )

        print(f"\nImage: {img_name}")
        print(f"Standard prediction count: {len(standard_prediction.object_prediction_list)}")
        print(f"SAHI sliced prediction count: {len(sliced_prediction.object_prediction_list)}")
        sliced_prediction.export_visuals(export_dir="out/sahi/", file_name=img_name, rect_th=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAHI implementation for YOLO with small dataset")
    parser.add_argument('--data', type=str, default='datasets/erick/data.yaml',
                        help='Path to data.yaml file')
    parser.add_argument('--output', type=str, default='datasets/sliced',
                        help='Output directory for the sliced SAHI dataset')
    parser.add_argument('--slice-size', type=int, default=640,
                        help='Size of the slices')
    parser.add_argument('--overlap-ratio', type=float, default=0.2,
                        help='Overlap ratio between slices')
    parser.add_argument('--skip-slicing', action='store_true',
                        help='Skip SAHI slicing')
    parser.add_argument('--skip-augmentations', action='store_true',
                        help='Skip Albumentations augmentations')

    args = parser.parse_args()

    # Create sliced (SAHI) dataset
    if not args.skip_slicing:
        data_yaml_path = create_sahi_dataset(
            args.data,
            args.output,
            args.slice_size,
            args.overlap_ratio
        )
    else:
        data_yaml_path = args.data

    if not args.skip_augmentations:
        print("Applying additional augmentations...")
        apply_albumentations_augmentations(data_yaml_path)

    print("Done!")

if __name__ == "__main__":
    main()