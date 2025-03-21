# Slot Detection for Agricultural Planting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eriklemy/slots_detection/blob/main/home_task.ipynb)

This project implements a robust object detection system for identifying planting slots in agricultural imagery, overcoming challenges of small object detection and limited dataset size. The complete technical process is documented in [home_task.ipynb](home_task.ipynb).

## Key Features
- **YOLOv8-based detection** optimized for small objects
- **Data augmentation** (3x dataset expansion)
- **SAHI integration** for high-resolution processing

## Project Overview

### Challenge
- Detect small planting slots in 4K+ resolution imagery, challenging when downscale to 640
- Limited initial dataset: 21 images total
- High variance in image resolutions and lighting

### Solution Strategy
1. **Data Enhancement**
   - SAHI-based image slicing (581 training patches)
   - Albumentations pipeline (1,940 final images)
   - Resolution balancing fixed to (640x640)

2. **Model Development**
   - YOLOv8n architecture selection
   - Transfer learning with frozen layers
   - Custom augmentation policies

3. **Performance Optimization**
   - mAP@50-95 metric monitoring
   - Sliced inference workflow
   - Confidence threshold tuning

## Technical Implementation
- **Framework**: Ultralytics YOLOv8
- **Hardware**: Google Colab (NVIDIA T4/V100)
- **Key Techniques**:
  - Mosaic augmentation (100% application)
  - MixUp blending (50% probability)
  - HSV color randomization

## Improvement Suggestions
- **Data**: Collect night/low-light scenarios, add occlusion cases
- **Model**: Experiment with larger models and architecture
- **Training**: Implement automated hyperparameter optimization (tunning)
- **Deployment**: Quantize model for edge devices

---

This version emphasizes outcomes over implementation details while maintaining key technical references. Readers are directed to the notebook for granular technical discussions and code examples.
---

### Extra:  
A simplified CLI version for training, slicing, and augmentation is available in the repository.  

**Usage Examples**:  
1. **Full pipeline** (slice + augment + train):  
   ```bash  
   python main.py --data /path/to/data.yaml --output /path/output --slice-size 640 --overlap-ratio 0.2  
   ```  

2. **Skip augmentations** (slice + train):  
   ```bash  
   python main.py --data /path/to/data.yaml --output /path/output --slice-size 640 --overlap-ratio 0.2 --skip-augmentations  
   ```  

3. **Skip slicing/augmentation** (train only):  
   ```bash  
   python main.py --data /path/to/data.yaml --output /path/output --slice-size 640 --overlap-ratio 0.2 --skip-augmentations --skip-slicing  
   ```  

**Defaults**:  
- `batch-size`: 8  
- `epochs`: 100  
- `img-size`: 640  
- `model-size`: n  

*For custom augmentations or advanced parameters, modify the configuration directly in the script.*  

Install dependencies if needed
```bash  
pip install -r requirements.txt
```  
---