# Fish Species Detection and Recognition

This repository contains a Jupyter Notebook for the detection and recognition of fish species using YOLOv11. The model leverages the **Fish Species Detection Dataset** from Kaggle, which contains images of various fish species, and applies object detection to classify and recognize different species in an underwater environment.

## Dataset

The **Fish Species Detection Dataset** is a meticulously curated resource, designed for developing and evaluating object detection models that specialize in identifying diverse fish species. It is intended to help advance computer vision applications in aquatic ecosystems through precise classification and detection techniques.

### Dataset Overview
- **Total Images**: 8,242 annotated images
- **Fish Species**: 13 species
  - AngelFish
  - BlueTang
  - ButterflyFish
  - ClownFish
  - GoldFish
  - Gourami
  - MorishIdol
  - PlatyFish
  - RibbonedSweetlips
  - ThreeStripedDamselfish
  - YellowCichlid
  - YellowTang
  - ZebraFish

The dataset is divided into:
- **Training Set**: 6,842 images
- **Validation Set**: 700 images
- **Test Set**: 700 images

#### Data Collection and Annotation
- **Source**: Images sourced from Roboflow and other online repositories
- **Annotation Format**: YOLOv11 format for compatibility with modern object detection frameworks
- **Pre-processing**: 
  - Auto-Orientation of images
  - Standardized resizing to 640x640 pixels
  - Augmented versions of images using rotations, random angle adjustments, exposure changes, and Gaussian blur

### [Download the Fish Species Detection Dataset from Kaggle](https://www.kaggle.com/datasets/mahmoodyousaf/fish-dataset)

## Model: YOLOv11n

In this project, we use **YOLOv11n** for object detection to classify and detect fish species in images. YOLOv11n is a highly efficient and lightweight model designed to work effectively on various computer vision tasks, including object detection in underwater environments.

### Installation

To run this notebook and train the model, follow the instructions below.

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fish-species-detection-and-recognition.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from Kaggle and place it in the appropriate folder.

4. Run the Jupyter Notebook to train and test the model on the fish species dataset.

### Notebook Overview

1. **Data Loading**: The dataset is loaded and preprocessed for training, including augmentation techniques to enhance model robustness.
   
2. **Model Training**: YOLOv11n is used for training the fish species detection model.

3. **Evaluation**: Model performance is evaluated using confusion matrices, precision-recall curves, and visualizations of detection results.

4. **Inference**: The trained model is used to predict fish species on test images, with the results displayed and saved for further analysis.

### Example Output

The following files are generated during model training and evaluation:
- **Confusion Matrix**
- **Precision-Recall Curves**
- **Model Results** (`results.png`, `P_curve.png`, `PR_curve.png`)
- **Visualizations of predictions on test images**

### Running the Inference Visualization

After training, the model can be run on the test set to visualize the detection of fish species. The bounding boxes around detected species are drawn on the images, and results are displayed and saved.

### Saving and Displaying Results

Images such as the **results**, **PR_curve**, **P_curve**, and the **predictions** are saved and displayed in the `/kaggle/working/inference_results` folder.

## Files in the Repository
- **notebook.ipynb**: The Jupyter notebook with the code for training and inference
- **requirements.txt**: A list of dependencies required for the project
- **inference_results/**: Directory to save and display images generated during model evaluation
