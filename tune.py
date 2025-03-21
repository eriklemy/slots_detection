import argparse
import yaml
from ultralytics import YOLO
import numpy as np
import torch

def main(args):
    # Configuration (to allow GPU use)
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Core parameters to tune
    tune_space = {
        'lr0': (1e-5, 1e-2),            # Initial learning rate
        'lrf': (0.01, 0.5),             # Final learning rate ratio
        'weight_decay': (1e-6, 1e-3),   # Optimizer weight decay
        'warmup_epochs': (3, 5),        # Warmup epochs
        'hsv_h': (0.0, 0.1),            # Image HSV-Hue augmentation
        'hsv_s': (0.5, 0.9),            # Image HSV-Saturation augmentation
        'hsv_v': (0.3, 0.7),            # Image HSV-Value augmentation
        'degrees': (30.0, 45.0),        # Image rotation
        'translate': (0.0, 0.3),        # Image translation
        'scale': (0.3, 0.5),            # Image scale
        'flipud': (0.5, 1.0),           # Horizontal flip probability
    }

    model = YOLO(args.model)

    if args.method == 'builtin':
        results = model.tune(
            data=args.data,
            space=tune_space,
            iterations=args.iterations,
            optimizer='AdamW',
            workers=8,
            plots=True,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            project=args.project,
            name=args.name,
            exist_ok=True,
            mixup=0.3,
            single_cls=True
        )
        
        # Save best configuration
        best_config = results.best_config
        with open(f'{args.project}/best_config.yaml', 'w') as f:
            yaml.dump(best_config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Hyperparameter Tuning")
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='yolov8n.yaml',
                       help='YOLO model configuration file')
    
    # Tuning method
    parser.add_argument('--method', choices=['builtin', 'rayTune'], default='builtin',
                       help='Tuning method to use')
    
    # Shared parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs per iteration')
    parser.add_argument('--batch', type=float, default=16,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--project', type=str, default='yolo_tuning',
                       help='Project name for saving results')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name')
    
    # Built-in tuning specific
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of tuning iterations (built-in method)')

    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)