# SignPose: A Dynamic Sign Language Detection Game
A LSTM Keypoint Model deployed on Flask that detects real time dynamic sign language (ASL animals) on browser.

Explore **AlphaSign**, the static version of our sign language game [here](https://github.com/yappeizhen/AlphaSign).


## Web Application Home Page
![signpose_cow](https://user-images.githubusercontent.com/69728128/140792190-7909b360-1703-4a62-8642-5fd6ab2400c1.JPG)


### Animal Dynamic Sign Language Detection

1. Cow

![cow_correct](https://user-images.githubusercontent.com/69728128/140792079-6e63fb65-9403-46a5-97d7-86bad5e25068.gif)

2. Elephant

![elephant_inference](https://user-images.githubusercontent.com/69728128/140793900-d1846730-643b-4daa-b39c-1d2f967caa22.gif)

3. Butterfly

![butterfly_inference](https://user-images.githubusercontent.com/69728128/140794314-463e5e06-a765-4234-9821-cda51ee477ef.gif)

4. Gorilla

![gorilla_inference](https://user-images.githubusercontent.com/69728128/140794693-b2abdc2e-8bb0-45b2-bf1f-eef7761c9c3a.gif)

5. Bird

![bird_inference](https://user-images.githubusercontent.com/69728128/140794935-ba3cc88c-df70-4e9d-ab90-6c46610b1d30.gif)

## Video Demo of Web App

https://user-images.githubusercontent.com/69728128/140792705-d215ba34-4cca-4d05-aa04-c8168f206cd4.mp4

## Exploratory Data Analysis
Breakdown of Train Test Split using Stratified Sampling (to ensure even distribution of train and test data)

![image](https://github.com/ngzhili/SignPose/blob/47cd40e2e2c0a842177228ab721211993c860188/readme-images/train-test-split.JPG)


## LSTM Model Architecture
```
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
```

## Model Evaluation

### Model Training Graphs
1. Train & Test Categorical Accuracy over epochs

![image](https://github.com/ngzhili/SignPose/blob/47cd40e2e2c0a842177228ab721211993c860188/readme-images/Model%20Training%20and%20Validation%20Categorical%20Accuracy.jpg)

2. Train & Test Loss over epochs (early stopping implemented to prevent overfitting)

![image](https://github.com/ngzhili/SignPose/blob/47cd40e2e2c0a842177228ab721211993c860188/readme-images/Model%20Training%20and%20Validation%20Loss.jpg)

### Confusion Matrix
1. Train Confusion Matrix
![image](https://github.com/ngzhili/SignPose/blob/47cd40e2e2c0a842177228ab721211993c860188/readme-images/train-confusion-matrix.JPG)

2. Test Confusion Matrix
![image](https://github.com/ngzhili/SignPose/blob/47cd40e2e2c0a842177228ab721211993c860188/readme-images/test-confusion-matrix.JPG)


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
