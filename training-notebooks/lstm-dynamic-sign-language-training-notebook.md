---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.1
  kernelspec:
    display_name: lstm_keypoint
    language: python
    name: lstm_keypoint
---

# LSTM Dynamic Sign Language Recognition Training Notebook


# 1. Import and Install Dependencies

```python
#!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib
```

```python
#!pip install mediapipe
```

```python
#!python -m ipykernel install --user --name=C:\Users\Zhili\.conda\envs\lstm_keypoint
```

```python
!pip install -r requirements.txt --user
```

```python
from platform import python_version
print(python_version())
```

```python
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
```

# 2. Visualizing Keypoints using MediaPipe (MP Holistic)

Reference for Google's Mediapipe API: https://google.github.io/mediapipe/solutions/holistic.html

```python
# Use Holistic Models for detections
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# Make keypoint detection, model can only detect in RGB
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB as model can only detect in RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Use Model to make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results): # draw landmarks for each image/frame
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
def draw_styled_landmarks(image, results): # draw landmarks for each image/frame, fix colour of landmark drawn
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
```

### 2.1 Open Computer Webcam using opencv

```python
# Use computer webcam
cap = cv2.VideoCapture(0)
# Set mediapipe model 

while cap.isOpened(): #open webcam
    # Read feed
    ret, frame = cap.read()
    # Show to screen
    cv2.imshow('OpenCV Feed: Hold Q to Quit', frame)
    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break
cap.release() #release webcam
cv2.destroyAllWindows()
```

### 2.2 Use Computer Webcam and make mediapipe keypoint detections

```python
# use computer webcam and make keypoint detections
cap = cv2.VideoCapture(0)

# Set mediapipe model configurations
min_detection_confidence = 0.5
min_tracking_confidence= 0.5

with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections by calling our function
        image, results = mediapipe_detection(frame, holistic) #mediapipe_detection(image, model) 
        #print(results)
        #print(results.face_landmarks)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Show to screen
        cv2.imshow('OpenCV Feed: Hold Q to Quit', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break
    cap.release() #release webcam
    cv2.destroyAllWindows()
```

```python
#show last frame with keypoints drawn using draw styled landmarks
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

```python
# call helper function to draw landmarks
draw_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

# 3. Extract Keypoint Values to be used for our model training

```python
#Show length of landmarks x,y,z spatial coordinates for right hand pose
len(results.right_hand_landmarks.landmark)
```

```python
# Show Results of landmark x,y,z spatial coordinates for face landmarks
print('Length of face landmarks:',len(results.face_landmarks.landmark))
print('Results Type:',type(results.face_landmarks))
print('Face landmarks Results:',results.face_landmarks)

# Convert facelandmarks to numpy array
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])if results.face_landmarks else np.zeros(468*3)
print(face)
```

```python
# Show Pose Connection Results
print('Pose Connection Results:',mp_holistic.POSE_CONNECTIONS)

# getting the landmarks x,y,z coordinates
pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)
print(pose)
```

```python
# Using list comprehension to extract landmark results if landmark for body part is detected, else replace it with a blank array of zeros of the same shape for each body part    
pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
```

```python
print(pose.shape)
#print(pose)
print(face.shape)
print(lh.shape)
print(rh.shape)
```

```python
# define extract keypoint function and convert to numpy array to be saved
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #return np.concatenate([pose, face, lh, rh]) # concatenate all the keypoints that are flattened
    return np.concatenate([pose, lh, rh])

result_test = extract_keypoints(results)
```

```python
extract_keypoints(results).shape
```

```python
# Total number of coordinates in results (pose, face, left hand and right hand)
33*4 + 468*3 + 21*3 + 21*3
```

```python
# Total number of coordinates in results (pose, left hand and right hand)
33*4 + 21*3 + 21*3
```

```python
# save results as numpy array
np.save('results', results)
```

```python
#load numpy array
np.load('results.npy')
```

```python
# Remove face landmarks from old training dataset
import numpy as np
index = [i for i in range(132,1536)]
actions = np.array(['Bird','Butterfly','Cow','Elephant','Gorilla','No Action'])

for action in actions:
    print(action) 
        # Loop through sequences aka videos
    for sequence in range(1,no_sequences+1): #range(30,60) # 30....59
         
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):
            
            #print(frame_num)
            npy_path = DATA_PATH+ '/'+action+'/'+str(sequence)+'/'+str(frame_num)
            test_np = np.load(npy_path+'.npy')
            new_np = np.delete(test_np, index)
            
            #print(test_np)
            #print(npy_path)
            #print(new_np.shape)
            # save keypoints to folder
            #np.save(npy_path, new_np)
```

# 4. Setup Folders for Collection of Keypoints for training LSTM model

```python
# Path for exported data, numpy arrays
#DATA_PATH = os.path.join('MP_Data')
DATA_PATH = os.path.join('test') 

# Actions that we try to detect
#actions = np.array(['Elephant'])
actions = np.array(['Gorilla'])
#actions = np.array(['Alligator'])
#actions = np.array(['Butterfly'])
#actions = np.array(['Cow'])
#actions = np.array(['Bird'])
#actions = np.array(['No Action'])
#actions = np.array(['Alligator','Butterfly','Cow','Elephant','Gorilla'])

# 60 videos worth of data for each action
no_sequences = 60

# Videos are going to be 30 frames in length (30 frames of data for each action)
sequence_length = 30

# 60 *30 = 1800 frames collected for each action
# each frame will contain 258 landmark values or 1662  landmark values

# for example
# bird folder
    # 0: 30 frames
    # 1: 30 frames
    # 2: 30 frames
    # 3: 30 frames
    #.
    #.
    #.
    #59: 30 frames

```

```python
#  loop through the actions we are detecting and make folders to store keypoints as numpy arrays
for action in actions: 
    #dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
    # loop through the sequences that we are collecting
    for sequence in range(1,no_sequences+1):
        try: #if the directory do not exist, we create a new directory to store the frames
            #os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
```

# 5. Collect Keypoint Values for Training and Testing

```python
#Capture Video using webcam
cap = cv2.VideoCapture(0)
# Set mediapipe model 
count = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(1,no_sequences+1): #range(30,60) # 30....59
            
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                
                # Read feed
                ret, frame = cap.read()
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic # Implement Collection Breaks between each sequence to allow me to reset and reposition to collect the action from start to finish
                if frame_num == 0: #If frame is 0, take a break
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500) #wait for 0.5 seconds
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                
                #print(npy_path)
                #print(keypoints.shape)

                # save keypoints to folder
                np.save(npy_path, keypoints)
                #cv2.waitKey(10)
                cv2.imwrite(f'test/{count}.jpg', image)
                count+=1
                
                
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
    cap.release()
    cv2.destroyAllWindows()
```

```python
import cv2
 
capture = cv2.VideoCapture(0)
 
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter('testvideo.avi', fourcc, 30.0, (640,480))
 
while (True):
    ret, frame = capture.read()
     
    if ret:
        cv2.imshow('video', frame)
        videoWriter.write(frame)
    if cv2.waitKey(1) == 27:
        break
        
capture.release()
videoWriter.release()
 
cv2.destroyAllWindows()
```

# 6. Data Preparation

```python
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import pandas as pd
```

```python
#Define Path to training data of numpy arrays
#DATA_PATH = os.path.join('MP_Data_bp_hp')
#DATA_PATH = os.path.join('MP_Data-6classes') # including facemesh

DATA_PATH = os.path.join('MP_Data-5classes') # including facemesh

# Define Model Run
run = 'run15'

#Define directory to save training graphs and confusion matrices
img_dir = f'Logs/{run}/images'

# create directory if image directory does not exist
if not os.path.exists(img_dir):
    os.makedirs(img_dir)    

# Define actions that we try to detect
actions = np.array(['Alligator','Butterfly','Cow','Elephant','Gorilla'])
#actions = np.array(['Bird','Butterfly','Cow','Elephant','Gorilla','No Action'])

# Sixty videos worth of data for each action
no_sequences = 60
# Videos are going to be 30 frames in length (30 frames of data for each action)
sequence_length = 30

#create label map dictionary
label_map = {label:num for num, label in enumerate(actions)} 
print(label_map)

#sequences represent x data, labels represent y data/the action classes.
sequences, labels = [], []
#Loop through the action classes you want to detect
for action in actions:
    #loop through each sequence
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# X = Training Data that contains spatial coordinates x,y,z of landmarks
X = np.array(sequences)

# y = categorical labels
y = to_categorical(labels).astype(int) #one-hot-encoding to catergorical variable

print('X Shape:',X.shape)
print('y Shape:',y.shape)

# train test split (95% train,5% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify=y)

print('X_train Shape:',X_train.shape)
print('y_train Shape:',y_train.shape)
print('X_test Shape:',X_test.shape)
print('y_test Shape:',y_test.shape)

# split imbalanced dataset into train and test sets with stratification
test_count_label = tf.reduce_sum(y_test, axis=0)
train_count_label = tf.reduce_sum(y_train, axis=0)

```

```python
# Show categorical list
actions.tolist()
```

```python
# Show categorical list
actions.tolist()
```

```python
left = pd.DataFrame(train_count_label,columns=['train_count'])
right = pd.DataFrame(test_count_label,columns=['test_count'])
df = left.join(right)

actions_list = actions.tolist()

left = pd.DataFrame(actions_list,columns=['class_names'])
df = left.join(df)
df
```

```python
left = pd.DataFrame(train_count_label,columns=['train_count'])
right = pd.DataFrame(test_count_label,columns=['test_count'])
df = left.join(right)

actions_list = actions.tolist()

left = pd.DataFrame(actions_list,columns=['class_names'])
df = left.join(df)
df
```

```python
# Plot Distribution of Train & Test Data after train_test_split
ax = df.plot.bar(x='class_names',rot=0)
ax.set_title('Distribution of Train & Test Data after train_test_split')
ax.figure.savefig(f'{img_dir}/train_test_distribution.jpg')
```

```python
# Plot Distribution of Train & Test Data after train_test_split
ax = df.plot.bar(x='class_names',rot=0)
ax.set_title('Distribution of Train & Test Data after train_test_split')
ax.figure.savefig('images/train_test_distribution.png')
```

# 7. Training LSTM Model

```python
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping

#Set up log directory to monitor training accuracy while training
log_dir = os.path.join('Logs/{}'.format(run))
tb_callback = TensorBoard(log_dir=log_dir)

model_dir = 'Logs/{}/model'.format(run)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

```

## 7.1 Build LSTM Neural Network using Keras

```python
del model
```

#### Experiment 1: 5 Classes (Facemesh, Hand pose and Body pose keypoints for training data)

```python
# Build LSTM Model Architecture Layers using Keras high-level # run 1
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
```

#### Experiment 2: 6 Classes (Facemesh, Hand pose and Body pose keypoints for training data)

```python
# Build LSTM Model Architecture Layers using Keras high-level  #run 2
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
```

#### Experiment 3: 6 Classes (Hand pose and Body pose keypoints for training data)

```python
# Build LSTM Model Architecture Layers using Keras high-level # experiment 3
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=False, activation='relu'))
#Dense layer with relu
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
```

```python
# Build LSTM Model Architecture Layers using Keras high-level # experiment 4
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
```

```python
# Build LSTM Model Architecture Layers using Keras high-level API # experiment 5
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(Dropout(0.2))
#Dense layer with relu
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
```

```python
# Build LSTM Model Architecture Layers using Keras high-level # experiment 6
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dropout(0.2))
#Dense layer with relu
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
```

```python
# Build LSTM Model Architecture Layers using Keras high-level # experiment 7
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
#Dense layer with relu
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
```

```python
# Build LSTM Model Architecture Layers using Keras high-level # experiment 8
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
```

```python
#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

### View Model Summary

```python
model.summary()
```

```python
import pydot
import pydotplus
from pydotplus import graphviz
from tensorflow.keras.utils import plot_model

plot_model(model, to_file=f'{img_dir}/model_plot.jpg', show_shapes=True, show_layer_names=True)
```

## 7.2 Define Model Parameters and Train Model

```python
epochs = 500
#checkpoint_dir = f"Logs/{run}/tmp/checkpoint"
model_filename = "Epoch-{epoch:02d}-Loss-{val_loss:.2f}.h5"
checkpoint_filepath = os.path.join('model/',model_filename)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'Logs/{run}/'+checkpoint_filepath,
    monitor='val_loss', #get the minimum validation loss
    mode='min',
    save_weights_only=True,
    save_best_only=True,
    verbose=2)

# Reference: https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
early_stopping =EarlyStopping(monitor='val_loss', patience=20,mode='auto',verbose=2)
```

```python
# Reference: https://www.tensorflow.org/guide/keras/train_and_evaluate
print(f"Fit model on training data for {epochs} epochs")
history = model.fit(
    X_train, y_train, 
    #batch_size=64,
    epochs=epochs,
    # We pass some validation data for monitoring validation loss and metrics at the end of each epoch
    validation_data=(X_test, y_test),
    verbose=2,
    batch_size=32,
    callbacks=[tb_callback, model_checkpoint_callback, early_stopping]
)

#https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
```

```python
# list all data in history
print(history.history.keys())
```

## 8. View Training Logs

```python
log_dir
```

```python
#%reload_ext tensorboard
```

```python
# Display Training Logs in Tensorboard 
%load_ext tensorboard
%tensorboard --logdir {log_dir}
```

## 9. Plot Training Results

```python
import pandas as pd
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

hist_df.to_csv(f'./Logs/{run}/history.csv',index = False)

df_hist = pd.read_csv(f'./Logs/{run}/history.csv')
hist_df
```

```python
min_loss_epoch = hist_df[hist_df['loss']==min(hist_df['loss'])].index.values
min_loss = min(hist_df['loss'])
print('Index of Minimum Loss =',min_loss_epoch[0])
print('Minimum Loss =',round(min_loss,2))
```

```python
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Training and Validation Categorical Accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{img_dir}/Model Training and Validation Categorical Accuracy.jpg')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Training and Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{img_dir}/Model Training and Validation Loss.jpg')
plt.show()
```

# 10. Load Best Model Weights/Checkpoint

```python
# Initialise Model again after deleting model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

```python
# Build LSTM Model Architecture Layers using Keras high-level # including facemesh #run 13
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

```python
# Build LSTM Model Architecture Layers using Keras high-level # run 12
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

```python
# Build LSTM Model Architecture Layers using Keras high-level # run 11
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

```python
# Build LSTM Model Architecture Layers using Keras high-level# run 10
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
#Dense layer with relu
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

```python
# Build LSTM Model Architecture Layers using Keras high-level# run 9
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=False, activation='relu'))
#Dense layer with relu
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

```python
# Build LSTM Model Architecture Layers using Keras high-level# run 8
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dropout(0.2))
#Dense layer with relu
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

```python
# Build LSTM Model Architecture Layers using Keras high-level API #run7
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(Dropout(0.2))
#Dense layer with relu
model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

```python
# Build LSTM Model Architecture Layers using Keras high-level (First Model)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

```python
actions = np.array(['Alligator','Butterfly','Cow','Elephant','Gorilla'])
```

```python
#model.load_weights(f'./Logs/{run}/model/Epoch-144-Loss-0.53.h5')
#model.load_weights(f'./Logs/{run}/model/Epoch-237-Loss-0.12.h5')
#model.load_weights(f'./Logs/{run}/model/Epoch-33-Loss-0.08.h5')
#model.load_weights(f'./Logs/{run}/model/Epoch-17-Loss-0.07.h5')

#model.load_weights(f'./Logs/run10/model/Epoch-13-Loss-0.05.h5')
#model.load_weights(f'./Logs/run11/model/Epoch-121-Loss-0.00.h5')
#model.load_weights(f'./Logs/run12/model/Epoch-38-Loss-0.12.h5')

#model.load_weights('./models/animal_asl_5_classes_1000_epoch_action.h5')

model.load_weights('./Logs/run15/model/Epoch-33-Loss-0.43.h5')
```

```python
del model
```

# 11. Make Predictions on X_test

```python
res = model.predict(X_test)

# Get y_predict and apply softmax function
np.argmax(res[4])
```

```python
# Predicted Action
actions[np.argmax(res[4])]
```

```python
# Actual Action
actions[np.argmax(y_test[4])]
```

```python
actions[y_test[1]]
```

# 12. Model Evaluation (Categorical Accuracy and Confusion Matrix)

Running these cells converts the predicition from their one-hot encoded representation to a categorical label e.g. 0,1 or 2 as opoosed to [1,0,0], [0,1,0] or [0,0,1].

```python
X_test.shape
```

```python
label_list = ['Alligator','Butterfly','Cow','Elephant','Gorilla']
label_list
```

```python
label_list = list(label_map.keys())
label_list
```

```python
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def generate_confusion_matrix_accuracy(X,y,types):
    predictions = model.predict(X)
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    print(f'{types} Overall Multiclassification Accuracy Score across all classes:',round(accuracy_score(y_true, y_pred),2))

    sns.heatmap(cm, xticklabels = label_list, yticklabels = label_list, annot = True, linewidths = 0.1, fmt='d',cmap='Blues') # cmap = 'YlGnBu')
    plt.title(f"{types} Confusion matrix", fontsize = 15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
generate_confusion_matrix_accuracy(X_train,y_train,'Train')
```

```python
generate_confusion_matrix_accuracy(X_test,y_test,'Test')
```

# 13. Run Real Time Keypoint Detection Model Using Webcam

```python
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

# Use Holistic Models for detections
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Make keypoint detection, model can only detect in RGB
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB as model can only detect in RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Use Model to make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results): # draw landmarks for each image/frame
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
    
def draw_styled_landmarks(image, results): # draw landmarks for each image/frame, fix colour of landmark drawn
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
# define extract keypoint function
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh]) # concatenate all the keypoints that are flattened
    #return np.concatenate([pose, lh, rh])

# Path for exported data, numpy arrays
#DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['Alligator','Butterfly','Cow','Elephant','Gorilla'])
#actions = np.array(['Bird','Butterfly','Cow','Elephant','Gorilla'])
#actions = np.array(['Bird','Butterfly','Cow','Elephant','Gorilla','No Action'])

# Thirty videos worth of data for each action
no_sequences = 30

# Videos are going to be 30 frames in length (30 frames of data for each action)
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)} #create label map dictionary

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# Build Model Architecture
#model = Sequential()
#model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
#model.add(LSTM(128, return_sequences=True, activation='relu'))
#model.add(Dropout(0.2))
#model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
#model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(actions.shape[0], activation='softmax'))

#model = Sequential()
#model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
#model.add(LSTM(128, return_sequences=True, activation='relu'))
#model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
#model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(actions.shape[0], activation='softmax'))

# Build LSTM Model Architecture Layers using Keras high-level # including facemesh #run 13
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# use categorical cross entropy as we are building a multiclass model
#metrics to track the accruacy of our model

#model_name = 'animal_asl_5_classes_1000_epoch_action'
#model.load_weights('models/{}.h5'.format(model_name))

#model_dir = 'Logs/run6/model'
#model_name = 'Epoch-209-Loss-0.00.h5'
#model.load_weights(f'{model_dir}/{model_name}')

#model.load_weights('./Logs/run13/model/Epoch-82-Loss-0.01.h5')
model.load_weights('./models/animal_asl_5_classes_1000_epoch_action.h5')
#model.load_weights('tutorial_weights_action.h5')

colors = [(245,221,173), (245,185,265), (146,235,193),(204,152,295),(255,217,179),(0,0,179)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1) #change length of bar depending on probability
        cv2.putText(output_frame, actions[num]+' '+str(round(prob*100,2))+'%', (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def add_image(image,results, action):
    #height,width = image.shape
    #print(image.shape)
    width = image.shape[1]#480
    height= image.shape[0]#640

    def overlay_transparent(background, overlay, x, y):
        
        # height and width of background image
        background_width = background.shape[1]
        background_height = background.shape[0]
        
        # if coordinate x and y is larger than background width and height, stop code
        if x >= background_width or y >= background_height:
            return background

        # height and width of overlay image
        h, w = overlay.shape[0], overlay.shape[1]

        #print('x:',x)
        #print('overlay_width:',w)
        #print('background_width:',background_width)
        #print('y:',y)
        #print('overlay_height:',h)
        #print('background_height:',background_width)

        if w >= background_width:
            return background
        if h >= background_height:
            return background
        
        # if coordinate x + width of overlay is larger than background width and height, stop code
        if x + w > background_width:
            #w = background_width - x
            #overlay = overlay[:, :w]
            return background
        if x - w < 2:
            #w = background_width - x
            #overlay = overlay[:, :w]
            return background
        if y + h > background_height:
            #h = background_height - y
            #overlay = overlay[:h]
            return background
        
        if y - h < 2:
            #h = background_height - y
            #overlay = overlay[:h]
            return background
        
        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
                ],
                axis = 2,
            )

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0

        background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

        return background

    index = 10
    
    face_keypoint=np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])if results.face_landmarks else np.zeros(468*3)
    #print(len(face_keypoint))
    #print(action)
    if face_keypoint.size != 0 and np.any(face_keypoint[index]) == True:

        if action =='Bird':
            file_name = './emoji/bird.png'
        elif action =='Butterfly':
            file_name = './emoji/butterfly.png'
        elif action =='Gorilla':
            file_name = './emoji/gorilla.png'
        elif action == 'Cow':
            file_name = './emoji/cow.png'
        elif action == 'Elephant':
            file_name = './emoji/elephant.png'
        elif action == 'Alligator':
            file_name = './emoji/alligator.png'
        else:
            file_name = './emoji/No_sign.png'
            
        if action != 'No Action':    
            overlay= cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

            #overlay= cv2.resize(overlay, (0,0), fx=min(0.1,float(1/face_keypoint[index][2]*-20)), fy=min(0.1,float(1/face_keypoint[index][2]*-20)))
            #print('z normalized',face_keypoint[index][2])
            #if face_keypoint[index][2]*-100 >1:
                #print('close to camera')
            #else:
                #print('far from camera')

            new_z = 0.1/((float(face_keypoint[index][2]*10)-(-1))/(1+1))
            #print('new_z',new_z)
            #print('z ',face_keypoint[index][2]*-10)
            #print('fx:',new_z)
            #print('fy:',new_z)

            #print(min(0.5,float(new_z)))

            overlay= cv2.resize(overlay, (0,0), fx=min(0.5,abs(float(new_z))), fy=min(0.5,abs(float(new_z))))

            #print('Normalized',face_keypoint[index])
            x = int(float(face_keypoint[index][0])*width)
            y = int(float(face_keypoint[index][1])*height)
            #print('Actual x',x)
            #print('Actual y',y)
            #cv2.circle(image,(x,y),3,(255,255,0),thickness= -1)

            #overlay = img2.copy()
            #image = cv2.rectangle(image, (x,y), (x+overlay.shape[1],y-overlay.shape[0]), (255,0,0), 3)

            #image = cv2.addWeighted(image,0.4,overlay,0.1,0)

            image = overlay_transparent(image, overlay, x - int(overlay.shape[0]/2), y-overlay.shape[0])


            #Setting the paste destination coordinates. For the time being, in the upper left
            #x1, y1, x2, y2 = x, y, overlay.shape[1], overlay.shape[0]

            #Synthetic!
            #image[y1:y2, x1:x2] = overlay[y1:y2, x1:x2]
```

```python
# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter('testvideo.avi', fourcc, 10, (640,480))

#import pafy
#import cv2
#url = "https://www.youtube.com/watch?v=jcnSKdZFr-M&ab_channel=ASLTHAT"
#video = pafy.new(url)
#best = video.getbest(preftype="mp4")
#cap = cv2.VideoCapture(best.url)
#cap = cv2.VideoCapture('./gif/combined.mp4')

# use computer webcam and make keypoint detections
cap = cv2.VideoCapture(0)

count = 0
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        
        # Read feed
        ret, image = cap.read()
        #if not ret:
            #break
        if ret == True:    
        #fps = cap.get(cv2.CAP_PROP_FPS)
        #print(fps)
        # for youtube video
        #width = 640
        #height = 480
        #dim = (width, height)
        #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            # Make detections
            image, results = mediapipe_detection(image, holistic)
            #print(results)
            #print(results.face_landmarks)
            #print(type(results.face_landmarks))

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            #if sequence[1] != None
            #print('face seq',sequence)
            
            # gets the most recent 30 frames
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        # Add_image
                        add_image(image, results, str(actions[np.argmax(res)]))
                        
                        if len(sentence) > 0: 
                            # if action is not in the last sentence, then we append the last action to the sentence
                            if actions[np.argmax(res)] != sentence[-1]: 
                                sentence.append(actions[np.argmax(res)])

                        else:
                            sentence.append(actions[np.argmax(res)])
                        
                    
                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
                #cv2.imwrite(f'{count}butterfly.jpg', image)
                #count +=1
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            
            #if str(actions[np.argmax(res)]) == 'Gorilla' and count >= 60:
                #break
            #cv2.imwrite(f'test/{count}.jpg', image)
            videoWriter.write(image)
            
            #count +=1
            # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            count+=1
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
```

```python
plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))
#image = cv2.imread('0butterfly.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

```python
plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))
#image = cv2.imread('0butterfly.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

```python
plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))
#image = cv2.imread('0butterfly.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

```python
plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))
#image = cv2.imread('0butterfly.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

```python
plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

```python
plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

```python
plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

```python
plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

```python
plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

## 13. Inference on Youtube and mp4 Video

```python
!pip install pafy
```

```python
!pip install youtube_dl
```

```python
# use computer webcam and make keypoint detections
#url = "https://www.youtube.com/watch?v=jcnSKdZFr-M&ab_channel=ASLTHAT"
#video = pafy.new(url)
#best = video.getbest(preftype="mp4")

#capture = cv2.VideoCapture(best.url)

capture = cv2.VideoCapture('gif/bird.mp4')

#capture = cv2.VideoCapture(0)
 
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter('bird_keypoint.avi', fourcc, 30.0, (1280, 720))

# Set mediapipe model 
min_detection_confidence = 0.5
min_tracking_confidence= 0.5
with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as holistic:
    while capture:
        # Read feed
        ret, frame = capture.read()
        
        if ret:
            #width = 640
            #height = 480
            height,width, channel = frame.shape
            dim = (width, height)
            #print(dim)
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            #print(frame.shape)

            # Make detections by calling our function
            image, results = mediapipe_detection(frame, holistic) #mediapipe_detection(image, model) 
            #print(results)
            #print(results.face_landmarks)

            # Draw landmarks
            draw_styled_landmarks(image, results)
        
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            videoWriter.write(image)
        else:
            break

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break
    capture.release() #release webcam
    videoWriter.release()
    cv2.destroyAllWindows()
```

## 15. Inference on Screen Capture

```python
!pip install mss
```

```python
!pip install Tkinter
```

```python
import ctypes
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
screensize

```

```python
import numpy as np
import cv2
from mss import mss
from PIL import Image

#bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}
bounding_box = {'top': 0, 'left': 0, 'width': screensize[0]//2, 'height': screensize[1]}

sct = mss()

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

#import pafy
#import cv2
#url = "https://www.youtube.com/watch?v=jcnSKdZFr-M&ab_channel=ASLTHAT"
#video = pafy.new(url)
#best = video.getbest(preftype="mp4")
#cap = cv2.VideoCapture(best.url)
#cap = cv2.VideoCapture('./gif/combined.mp4')



count = 0
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        sct_img = sct.grab(bounding_box)
        

        image = np.array(sct_img)
        # Read feed
        #ret, image = cap.read()
        #if not ret:
            #break
        #if ret == True:    
        #fps = cap.get(cv2.CAP_PROP_FPS)
        #print(fps)
        # for youtube video
        #width = 640
        #height = 480
        #dim = (width, height)
        #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        # Make detections
        image, results = mediapipe_detection(image, holistic)
        #print(results)
        #print(results.face_landmarks)
        #print(type(results.face_landmarks))

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        #if sequence[1] != None
        #print('face seq',sequence)

        # gets the most recent 30 frames
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            #print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    # Add_image
                    add_image(image, results, str(actions[np.argmax(res)]))

                    if len(sentence) > 0: 
                        # if action is not in the last sentence, then we append the last action to the sentence
                        if actions[np.argmax(res)] != sentence[-1]: 
                            sentence.append(actions[np.argmax(res)])

                    else:
                        sentence.append(actions[np.argmax(res)])


            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            #cv2.imwrite(f'{count}butterfly.jpg', image)
            #count +=1
        # Show to screen
        #cv2.imshow('OpenCV Feed', image)
        cv2.imshow('screen capture', image)

        #if str(actions[np.argmax(res)]) == 'Gorilla' and count >= 60:
            #break
        #cv2.imwrite(f'test/{count}.jpg', image)
        #videoWriter.write(image)
        
        #count +=1
        # Break gracefully
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
        count+=1
            
cv2.destroyAllWindows()
```

```python
import cv2
import numpy as np
import cv2
from mss import mss
from PIL import Image
from PIL import ImageGrab

#bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}
bounding_box = {'top': 0, 'left': 0, 'width': screensize[0], 'height':screensize[1]}

sct = mss()

#capture = cv2.VideoCapture(0)
 
#fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
#videoWriter = cv2.VideoWriter('video_demo_lstm.avi', fourcc, 30.0, (640,480))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('outofframe.avi', fourcc, 10, (screensize[0],screensize[1]-60))


while True:
        
    img = ImageGrab.grab(bbox=(0,0,screensize[0],screensize[1]-60)) #bbox specifies specific region (bbox= x,y,width,height)
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow("test", frame)
    
    #sct_img = sct.grab(bounding_box)
    #print(sct_img)
    #print(sct_img)
    #print(sct_img.size)  
    #image = np.array(sct_img)
    #image = cv2.resize(image, (640, 480))
    #print(image.size)
    
    #cv2.imshow('video', image)
    videoWriter.write(frame)
    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break
        
#capture.release()
videoWriter.release()
 
cv2.destroyAllWindows()
```
