import os
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
from ultralytics import YOLO
import albumentations as A
from tqdm import tqdm

import torch
import utils

print(f"CUDA available: {torch.cuda.is_available()}")
device = 0 if torch.cuda.is_available() else 'cpu'

def train_yolo_model(data_yaml_path, model_size='n', epochs=100, batch_size=16, img_size=640):
    """
    Train a YOLO model on the dataset.
    
    Args:
        data_yaml_path: Path to the data.yaml file
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of epochs to train
        batch_size: Batch size
        img_size: Image size
    """
    model = YOLO(f'yolov8{model_size}.pt')
    
    augmentation = {
        "augment": True, # Allow TTA (Test Time Augmentation)
        "mosaic": 1.0,
        "mixup": 0.5,
        "scale": 0.5,
        "fliplr": 0.5,
        "flipud": 0.2,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.8,
    }
    
    # delete old folder if exists instead of overwriting or writing a new one
    shutil.rmtree("slots/exp", ignore_errors=True)
    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        workers=2, 
        freeze=list(range(11)), # 0 ... 10
        patience=30,            # early stop
        device=device,
        project='slots',
        name=f'exp',
        seed=42,
        **augmentation,
    )
        
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SAHI implementation for YOLO with small dataset")
    parser.add_argument('--data', type=str, default='datasets/erick/data.yaml', 
                        help='Path to data.yaml file')
    parser.add_argument('--output', type=str, default='datasets/erick_sahi', 
                        help='Output directory for the SAHI-augmented dataset')
    parser.add_argument('--slice-size', type=int, default=640, 
                        help='Size of the slices')
    parser.add_argument('--overlap-ratio', type=float, default=0.2, 
                        help='Overlap ratio between slices')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='Image size')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training the YOLO model')
    parser.add_argument('--skip-slicing', action='store_true', 
                        help='Skip SAHI slicing (only apply augmentations)')
    parser.add_argument('--skip-augmentations', action='store_true', 
                        help='Skip Albumentations augmentations')
    
    args = parser.parse_args()
    
    # Create SAHI-augmented dataset
    if not args.skip_slicing:
        data_yaml_path = utils.create_sahi_dataset(
            args.data, 
            args.output, 
            args.slice_size, 
            args.overlap_ratio
        )
    else:
        data_yaml_path = args.data
    
    if not args.skip_augmentations:
        print("Applying additional augmentations...")
        utils.apply_albumentations_augmentations(data_yaml_path)
    
    if not args.skip_training:        
        print(f"Training YOLOv8{args.model_size} model...")
        results = train_yolo_model(
            data_yaml_path,
            model_size=args.model_size, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            img_size=args.img_size
        )
    
        utils.plot_image_results("slots/exp/")
        print(f"Evaluating model with SAHI...")
        utils.evaluate_with_sahi(f"slots/exp/weights/best.pt", data_yaml_path)
    print("Done!")

if __name__ == "__main__":
    main()