# Landslide Detection Project

## Introduction
In this project, we demonstrate how to set up and implement a machine learning-based system for predicting landslides using various features like RGB images, NDVI, DEM, and Slope. This guide walks you through collecting the dataset, setting up the environment, preprocessing the data, training the model, and executing the prediction code.

## Key Steps to Initiate the Project

### Step 1. Create an Environment
Open the terminal and execute the following commands to set up the environment.

```bash
# Create a new environment
conda create --name landslide-env python=3.8

# Activate the environment
conda activate landslide-env
# Navigate to the landslide-detection directory
cd path\to\landslide-detection
```
### step 2. Collect the Dataset
Gather a dataset consisting of images representing landslide-prone areas. Ensure you have sufficient data for various features to train an effective model.
### step 3. Set Up Directory Structure
Organize your dataset into train, test, and validation sets. Place these folders in the data directory.
```bash
landslide-detection/
└── data/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── test/
    │   ├── images/
    │   └── masks/
    └── validation/
        ├── images/
        └── masks/
```
### step 4. Preprocess and Augment the Data
Use scripts to preprocess and augment your dataset as needed. Ensure the files are saved in the correct directories.
### Step 5. Train the Model
Use the following command to start training the model with the labeled dataset.
```bash
python train.py --img 640 --batch 16 --epochs 50 --data ./data.yaml --weights yolov5s.pt --cache
```
This command trains the model using the labeled dataset and saves the trained model.
### Step 6. Save the Trained Model
Once training is complete, download the trained model and save it in the root directory of the project.
### Step 7. Configure Custom Dataset
Update the configuration files to specify the number of classes and paths to your train, test, and validation datasets.
### Step 8. Modify Paths in Scripts
In the relevant scripts (e.g., detect.py), update the paths to your test, train, and validation datasets.
### Step 9. Execute the Prediction Code
Run the following command to execute the detection code:
```bash
python detect.py --source your_source_path --weights trained_model.pt --conf 0.40 --save-txt --view-img
```
Replace your_source_path with the appropriate path.
### Sample Code Snippets
#### Loading Datasets:
```bash
import h5py
with h5py.File('data/images.h5', 'r') as img:
    images = img['images'][:]
```
#### Model Training:
```bash
model.fit(train_data, epochs=50, validation_data=val_data)
```
### Conclusion
By following these steps, you can set up a comprehensive landslide detection system using machine learning techniques. This system aims to enhance prediction accuracy and support real-time analysis in landslide-prone areas.


