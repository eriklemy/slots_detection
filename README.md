# slots_detection
This project implements an object detection system to detect Slots for planting flowers. Despite the limited dataset size, we explore techniques to develop a detection model, that can work properly.

### Project Objectives
1. Train an object detection model with the data (Yolo or any other model).
2. Evaluate the model performance (select and justify evaluation metrics).
3. Discuss the model performance and errors.
4. Discuss the steps used to solve the problem.
5. Suggest improvements in data/model/etc.

### Dataset Characteristics
- Classes: 1 (Slots)
- Total images: 21 
- split: 
    - train: 16 images (75%)
    - val: 5 images    (25%)

### Technical Environment
- Hardware: Google Colab GPU (NVIDIA T4/V100)
- Framework: Ultralytics YOLO
- Acceleration: CUDA-enabled training

### **Steps**
1. Dataset Evaluation
2. Image Analysis
3. Train Model
4. Improvement Strategy
5. Results Discution

--- 
## 1. Dataset Verification

### **Pre-Training Dataset Evaluation**  

Before training, it is essential to evaluate the dataset by verifying that both images and labels are correct. This process also helps in identifying a suitable model based on the dataset format, reducing the need for significant modifications if it already aligns with a specific model.  

### **Reasons:**  

1. **Prevent Training Failures**  
   - YOLO coordinates outside the [0,1] range can cause numerical errors (e.g., NaN loss).  
   - Incorrect folder structure can prevent data from loading properly.  

2. **Ensure Annotation Quality**  
   - Poor annotations can reduce mAP scores.  
   - Unlabeled objects introduce bias, leading to inaccurate learning.  

3. **YOLO Compatibility**  
   - Requires a specific directory format:  
     ```bash
     dataset/
     ├── train/images/  # .jpg, .png
     └── train/labels/  # .txt (one file per image)
     ```  
   - Class IDs must match the YAML configuration file.  

The provided dataset follows the standard structure, with separate training and validation sets, as well as distinct folders for images and labels. Since it includes a YAML file and annotations in `.txt` format structured as:  
   - (`class`, `center_x`, `center_y`, `height`, `width`)  
   - Values normalized between **0 and 1**  

This suggests that the dataset was created using an auxiliary labeling tool, such as [Roboflow](https://roboflow.com), and later converted to YOLO format. By visualizing an image alongside its corresponding annotations, the format was confirmed.**negrito**

---
## 2. Image Analysis
### **Visualizing Image Resolutions**  

Analyzing the distribution of image resolutions in the **training** and **validation** sets helps identify potential imbalances that could affect model training and generalization.  

### **Methodology:**  
1. **Extract Resolutions:** Collect width and height from all images.  
2. **Count Occurrences:** Determine the number of unique resolutions and the number of images per resolution.  

### **Observations:**  

1. **Training Set Imbalance:**  
   - 80% of the training images are concentrated in just **two resolutions** (`4000x2252` and `4032x3024`).  
   - Resolutions like `1600x900` and `2000x1126` contain only **one image each**, which could lead to *overfitting* for these specific formats.  

2. **Limited Validation Coverage:**  
   - The validation set covers only **50%** of the resolutions found in the training set.  
   - The resolution `4032x3024` has only **one image** in validation, making evaluation unreliable for this format.  

### **Implications:**  
- **Resolution Bias:** The model may perform poorly on underrepresented resolutions (e.g., `1600x900`).  
- **Preprocessing Needs:** Different resolutions require strategies like *resizing* or *padding* for consistency.  
- **Generalization Issues:** The validation set does not sufficiently cover the training resolutions, increasing performance drops in real-world deployment.

### Analyze Color Channels

Comparing the statistical distributions (mean and standard deviation) of the color channel intensities (BGR) between the training and validation sets to ensure consistency and identifying potential biases that may affect the model.

### **Methodology:**  
1. **Calculation of Statistics:**  
   - Mean and standard deviation of the average intensities for each channel (Blue, Green, Red) across all images.  
2. **Visualization:**  
   - Histograms to compare intensity distributions.  
   - Boxplots to analyze variation and outliers.  

### **Observations:**
The original dataset does not exhibit many outliers and appears consistent with the expected scenario. The high Green and Red values are due to the presence of plants and ground textures, while Blue is typically less prominent in natural environments. As a result, there is no significant indication of bias. However, applying a Hue augmentation could help simulate greater color variability and further enhance the model's robustness.

---
## 3.Train Model

### Why YOLO?
The YOLO architecture was selected for its:

Real-time inference capabilities
Balance between accuracy and speed
Strong performance on small datasets
Extensive pre-trained weights availability
Dataset Compatibility
Agricultural Applications [1, 2, 3, 4, 5]
SAHI compatibility [1, 2]
Augmentation capabilities
Training Parameters
With the model defined, the training starts! First, there are a few important parameters to decide on.

### Model architecture & size (model):

Several YOLO versions exists, like yolov5, yolov8 and yolov11 with different models sizes (yolov8n.pt, yolo8m.pt, yolo11n.pt, yolo11m.pt, yolo11x.pt). Larger models run slower but have higher accuracy, while smaller models run faster but have lower accuracy. For this project the yolov8n.pt was chosen since it has less params than older models and provides balanced trade-off between speed and accuracy compared to others models like v9 (more computational heavy) and YOLO11 with is optimized for speed and efficiency, is not the best choice for this task since its focus is on detection smaller objects in higher resolution imagens. The decision to use the n variant was made to minimize/prevent overfitting during the transfer learning while ensuring that the model remains sufficiently capable of handling the complexity of small object detection.

### Model architecture & size (model):

Several YOLO versions exist, such as YOLOv5, YOLOv8, and YOLOv11, each offering models of varying sizes (yolov8n.pt, yolov8m.pt, yolo11n.pt, yolo11m.pt, yolo11x.pt). Larger models tend to run slower but offer higher accuracy, while smaller models run faster but sacrifice some accuracy. For this project, the yolov8n.pt model was selected. Although it has fewer parameters than the newer models (except YOLOv11), YOLOv8 provides a balanced trade-off between speed and accuracy. YOLOv11, while optimized for speed and efficiency, is not the best choice for this task since its focus is on maximizing performance in real-time scenarios, which is not the primary requirement here. YOLOv8, on the other hand, is well-suited for achieving good detection accuracy with reasonable processing time, making it ideal for the dataset and the computational constraints of this project.

### Number of epochs (epochs) and (batchs)

With a limited dataset, an initial setting of 50 epochs is used to provide sufficient training while mitigating the risk of overfitting. YOLO automatically adjusts the batch size based on GPU memory, typically utilizing around 60% of the GPU’s RAM. For a resolution of 640x640, this usually results in a batch size between 8 and 16. This configuration strikes a balance between efficient training and resource utilization.

### Resolution (imgsz)

Image Resolution has a large impact on the speed and accuracy of the mode. Lower resolution model will have higher speed but less accuracy. YOLO models are typically trained and inferenced at a 640x640 resolution.

### Data Augmentation

Given the limited size of our dataset, we apply data augmentation to improve the model's generalization and robustness. YOLO provides built-in augmentation tools that eliminate the need to create entirely new datasets. The following augmentation techniques are in use:

Mosaic (1.0) – Set to 1.0 to ensure that all images will have that augmentation. This augmentation combines multiple images into one, helping the model learn object variations and different backgrounds.
MixUp (0.5) – Set to 0.5 to ensure that this augmentation only happens 50% of the time. This augmentation blends two images and their labels, improving the model's ability to handle occlusions and uncertainties reducing overfitting.
Scale (0.5) – Randomly resizes objects to enhance scale invariance. (50% probability to scale)
Flip (fliplr: 0.5, flipud: 0.2) – Applies horizontal and vertical flips to introduce viewpoint variations, horizontal is particularly effective for overhead/aerial images like those in our datase. (Aplied to 50% and 20% of images)
HSV Augmentation (hsv_h: 0.015, hsv_s: 0.9, hsv_v: 0.9) – Slightly hye Adjust to prevent extreme color shift, high saturation, and brightness to improve color invariance.
Transfer Learning

Using a pre-trained model leverages knowledge acquired from large datasets, facilitating the extraction of relevant features even if the target class isn't present in the original model. Freezing some of the initial layers preserves fundamental features, while the upper layers are fine-tuned to learn the new classes and adapt to our specific domain.

## Model Evaluation

To evaluate the model's performance, the metrics **Precision**, **Recall**, **mAP** (Mean Average Precision), and **mAP@50-95** were selected. These metrics are widely used in object detection evaluations and are standard in frameworks like YOLO, as well as in renowned benchmarks such as COCO.

**Precision** and **Recall** provide essential insights into the quality of the detections by indicating the proportion of true positives relative to predictions and the model's ability to correctly identify objects. In contrast, **mAP@50-95** assesses the average precision of the model across different IoU (Intersection over Union) thresholds, offering a comprehensive view of detection accuracy at varying levels of overlap between predictions and ground truth objects.


## 4. Improvement Strategy
## Slicing Strategy for Improving Small Object Detection

The original images had high resolutions (e.g., 4032x3024, 4000x2252), and the objects to be detected were very small relative to the overall image size:  
- **Average BBox area:** 18,936.92 pixels²  
- **Median BBox area:** 20,804.56 pixels²  

For reference, in a typical 4000x2252 image (~9MP):  
- The object occupied only **~0.2% of the total image area** (18,936 / 9,000,000 ≈ 0.0021).

### Identified Problem  
At such high resolutions, the object becomes **nearly imperceptible** to conventional detection models, even when using absolute bounding boxes. This results in:  
1. Loss of detail during downsampling (resizing to standard resolutions such as 640x640).  
2. Learning difficulties due to the low pixel density of the object.

### Implemented Solution  
The images were divided into **640x640 pixel patches** using a slicing technique, resulting in:  
- **Expansion of the training dataset:** from 16 to 581 images.  
- **Benefits:**  
  - **Artificial increase in the object's relative resolution:**  
    In a 640x640 patch (409,600 pixels²), the object's area now represents **~4.6%** (18,936 / 409,600) — making it 22 times more prominent.  
  - Reduced computational complexity, as there are fewer pixels to process per iteration.  
  - Generation of pseudo-augmented data that provides contextual variations.

### Using SAHI (Slicing Aided Hyper Inference) for Small Objects

#### How SAHI Complements YOLO
| **Approach**         | **SAHI Advantage**                   | **Limitation of Pure YOLO**      |
|----------------------|--------------------------------------|----------------------------------|
| **Inference**        | Processes the image in slices        | Processes the entire image       |
| **Object Size**      | Detects objects smaller than 50px    | Often misses smaller objects |
| **Computational Cost** | Increases processing time by ~20%, but improves mAP@50-95 by ~30% | Faster inference, but with lower precision |

This slicing strategy, especially when paired with SAHI, significantly enhances the detection of small objects by increasing their relative size and preserving more details during inference.

### Data Augmentation with Albumentations

The use of the Albumentations library, in conjunction with YOLO augmentations, allows for the generation of a larger and more robust dataset by providing techniques not present in YOLO or with limited configuration options. In this sense, the following augmentations were applied to the pipeline:


  ```python
    # Augmentation pipeline
    aug_transforms = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.RandomGamma(gamma_limit=(60, 115), p=0.3),
        A.GaussNoise(p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
  ```

# 5. Results Discution

The levels of brightness, saturation, hue and GaussNoise were kept at their default values and adjusting just the probability to happen to , as they are already tuned for use with YOLO, while the RandomGamma required fine-tuning to better distribute the irregular exposure generated (dark/light). The use of Albumentations allowed for a 3x increase in the dataset size, totaling 1,940 images, with 1,743 for training and 197 for validation. 

As can be seen in the figure bellow (after running the block), applying SAHI significantly improved class identification accuracy, increasing from approximately 50% to 80%. Additionally, detections became more consistent, reducing false positives and further demonstrating the effectiveness of the technique. However, it was observed that images with better lighting conditions yielded more stable results, suggesting that illumination plays a crucial role in detection performance and can be further improved.

## Further Improvement Sugestions
1. **Data Improvements**:
   - Increase dataset size
   - Add more challenging scenarios (occlusions, varying lighting)
   - Improve annotation quality to reduce label incosistence
2. **Model Improvements**:
   - Larger YOLO variants (yolov8s/yolov8x/yolo11s/yolo11l)
   - Explore R-CNN models for higher accuracy (like detectron2)
3. **Training Improvements**:
   - Hyperparameter optimization (learning rate, weight decay)
   - Tunning with a wide space range (Computational Heavy)
   - Longer training with progressive resizing
4. **Post-processing**:
   - Optimize confidence thresholds to reduce false positives
   - Implement/Use test-time augmentation (TTA)