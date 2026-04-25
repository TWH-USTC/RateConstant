# Overview

This project aims to predict molecular rate constants using Graph Neural Network (GNN) and Gaussian Process Regression (GPR) algorithms.


## 1. Descriptions of the python scripts.
Below is the detailed description of the code structure and module functionalities.

### 1.1 Data Preparation

All raw data and related scripts should be placed in the directory named 'Dataset'.

(1) data_combination.py

This script is used to combine data collected from different literature sources.

(2) data_preprocess.py

This script handles the cleaning and formatting of the combined dataset.

(3) fingerprints.py

This script generates molecular fingerprints to serve as input features for traditional machine learning models.

(4) get_sdf_files.py

This script generates SDF files, which serve as inputs for the UniMol2 model to produce pre-trained molecular embeddings.

### 1.2 Fingerprint-based Machine Learning Model (baseline model)

The relevant code is located in the Code/ML directory.

(1) ml.py

The core script for training and evaluating traditional machine learning models based on molecular fingerprints.

(2) few_shot.py

Used to evaluate the few-shot learning capabilities of the models.

### 1.3 GNN and GPR algorithms based model

The relevant code is located in the Code/DL directory.
#### GNN
(1) main.py

Main script for training and evaluating the GNN model.

(2) model.py

Defines the GNN architecture.

(3) construct_dataset.py

Constructs the dataset used for training and evaluation.

(4) get_embeddings_unimol.py

Retrieves pre-trained molecular embeddings using the UniMol2 model.

(5) get_embeddings.py

Extracts molecular representations learned by the GNN model.

(6) predict.py

Predicts rate constants for new molecules using the trained GNN model.

(7) few_shot.py

Evaluates the few-shot learning capability of the GNN model.

(8) data.py

Define PyTorch dataset wrapper class.

(9) featurization.py

Contains code for constructing molecular graphs based on SMILES.

#### GPR

The corresponding code is located in the Code/DL/GP directory.

(10) main.py

Main script for training and evaluating the GPR model.

(11) predict.py

Obtains rate constants (mean) and predictive uncertainty (standard deviation) for new molecules using the trained GPR model.

(11) GP.py

Defines the GPR model architecture.


## 2. Instructions for use

Below is the detailed description to run the codes. The running time was estimated according to our hardware. The hardware configuration consisted of dual Intel Xeon Gold 5220R processors (2.20 GHz; 48 cores/96 threads) and an NVIDIA GeForce GTX 1650 GPU (4 GB).

### 2.1 Preprocess the dataset

(1) Combine data from multiple sources and generate a TXT file containing the merged dataset, taking about 3 hours.

    python Dataset/data_combination.py

(2) Preprocess the merged dataset, taking about several seconds.

    python Dataset/data_preprocess.py

### 2.2 Prepare pretained molecule embeddings for GNN model

(1) Obtain SDF files of molecules according to their SMILES for pretained Unimol2 model, taking about 10 minutes.

    python get_sdf_files.py

(2) Get Unimol2 embeddings for molecules according to their SMILES, taking about 10 minutes.

    python Code/DL/get_embeddings_unimol.py

### 2.3 Train and evaluate GNN model

(1) Split dataset. Generate the CSV files containing the training and test set, taking about several seconds.

    python Code/DL/construct_dataset.py
(2) Train and evaluate the GNN model. Generate TXT files containing the measured and predicted values of pollutants in training and test set, taking about 5 hours.

    python Code/DL/main.py
(3) Evaluate the few-shot learning ability of the GNN model, taking about 12 hours.

    python Code/DL/few_shot.py
(4) Predict the rate constant of typical pollutants, taking about several seconds.

    python Code/DL/predict.py

### 2.4 Train and evaluate the GPR model

(1) Obtain embeddings using the trained GNN model, taking about several minutes. This step should be craaied out after fininshing the training of the GNN model.

    python Code/DL/get_embeddings.py
(2) Train and evaluate the GPR model, taking about several minutes.

    python Code/DL/GP/main.py
(3) Obtain the predicted rate constants and uncertainty of typical pollutants, taking about several seconds.

    python Code/DL/GP/predict.py

### 2.5 Train and evaluate the baseline model

(1) Obtain fingerprints for the ML model, taking about several minutes.

    python Dataset/fingerprints.py
(2) Train and evaluate the ML model, taking about 0.5 hour.

    python Code/ML/ml.py
(3) Evaluate the few-shot learning ability of the ML model, taking about 3 hours.

    python Code/ML/few_shot.py


## 3. Demo

python Dataset/data_combination.py
python Dataset/data_preprocess.py
python get_sdf_files.py
python Code/DL/get_embeddings_unimol.py
python Code/DL/construct_dataset.py
python Code/DL/main.py
python Code/DL/get_embeddings.py
python Code/DL/predict.py
python Code/DL/GP/main.py
python Code/DL/GP/predict.py

The complete demo would finish runing in 24 hours on our hardware. The excepted outputs are predicted rate constants and uncertainty of pollutants.


## 4. Installation guide

All experiments were conducted on Ubuntu 20.04 LTS. The implementation does not depend on operating-system-specific functions and can be run on other platforms with compatible Python and package dependencies.

The virtual environment was configured using Anaconda3.


conda create -n <env_name> python=3.9

pip install cirpy==1.0.2

pip install rdkit==2022.9.5

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install unimol-tools==0.1.4.post1

pip install scikit-learn==1.6.1

pip install gpytoch==1.13
