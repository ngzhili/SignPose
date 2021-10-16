# LSTM_Keypoint_Sign_Language_Detector
An LSTM Keypoint Model that detects real time dynamic sign language deployed on flask API.

## Exploratory Data Analysis
Breakdown of Train Test Split using Stratified Sampling (to ensure even distribution of train and test data)

![image](https://github.com/ngzhili/LSTM_Keypoint_Sign_Language_Detector/blob/4dcd3fb656f62611dc81b497ab7eef885ff3ab4a/readme-images/train_test_distribution.png)

## Model Evaluation

### Model Training Graphs
1. Train & Test Categorical Accuracy over epochs

![image](https://github.com/ngzhili/LSTM_Keypoint_Sign_Language_Detector/blob/c46727ba2c1eba3ce4712fb839539bccfb874811/readme-images/run4%20Model%20Training%20and%20Validation%20Categorical%20Accuracy.jpg)

2. Train & Test Loss over epochs

![image](https://github.com/ngzhili/LSTM_Keypoint_Sign_Language_Detector/blob/4dcd3fb656f62611dc81b497ab7eef885ff3ab4a/readme-images/run4%20Model%20Training%20and%20Validation%20Loss.jpg)

### Confusion Matrix
1. Train Confusion Matrix
![image](https://github.com/ngzhili/LSTM_Keypoint_Sign_Language_Detector/blob/4dcd3fb656f62611dc81b497ab7eef885ff3ab4a/readme-images/run4%20Train%20Confusion%20Matrix.jpg)

2. Test Confusion Matrix
![image](https://github.com/ngzhili/LSTM_Keypoint_Sign_Language_Detector/blob/4dcd3fb656f62611dc81b497ab7eef885ff3ab4a/readme-images/run4%20Test%20Confusion%20Matrix.jpg)


## Installation in command prompt
### 1. Clone Repo
```
git clone https://github.com/ngzhili/LSTM_Keypoint_Sign_Language_Detector.git
```
### 2. Create your virtual environment (venv) using conda
```
conda create -n <VENV_NAME> python=3.8
```

### 3. Activate your virtual environment (venv) using conda
```
conda activate <VENV_NAME>
```

### 4. Install dependencies in venv
```
pip install -r requirements.txt
```

### 5. Change directory to Sign-Language-Image-Recognition
```
cd <PATH/TO/Sign-Language-Image-Recognition>
```
### 6. Run Application in same directory as app.py
```
flask run
```
