from flask import Flask,render_template,Response,request,json,jsonify
import cv2
import time
import sys

import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.keras.callbacks import TensorBoard

sys.path.append('./mediapipe_functions.py')
from mediapipe_functions import mediapipe_detection, draw_landmarks, draw_styled_landmarks, extract_keypoints, add_image, prob_viz

video_id = 'no'

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    #if request.method == 'POST':
        #if request.form.get('LSTM_model') == 'start_detection':
            #print('hello123')
            #video_id = 'lstm'

    """Video streaming home page."""
    return render_template('index.html')

# Initialise detection confidence
lstm_threshold = 0.5
toggle_keypoints = True

@app.route('/process_slider_value', methods=['POST', 'GET'])
def process_slider_value():
    global lstm_threshold
    global toggle_keypoints
    if request.method == "POST":
        slider_data = request.get_json()
        print(slider_data)
        #print(slider_data[0]['slider'])
        lstm_threshold = float(slider_data[0]['slider'])
        #toggle_keypoints = slider_data[0]['toggle']
        print('LSTM Detection Threshold:',lstm_threshold)
        #print('Toggle_keypoints:',toggle_keypoints)
    
    results = {'processed': 'true'}
    return jsonify(results)

@app.route('/process_toggle_value', methods=['POST', 'GET'])
def process_toggle_value():
    global toggle_keypoints
    if request.method == "POST":
        toggle_data = request.get_json()
        print(toggle_data)
        #print(slider_data[0]['slider'])
        toggle_keypoints = toggle_data[0]['toggle']
        print('Toggle_keypoints:',toggle_keypoints)
    
    results = {'processed': 'true'}
    return jsonify(results)


# Use Holistic Models for detections
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Actions that we try to detect
#actions = np.array(['hello', 'thanks', 'iloveyou'])
actions = np.array(['Alligator','Butterfly','Cow','Elephant','Gorilla'])

label_map = {label:num for num, label in enumerate(actions)} #create label map dictionary

# Build Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#Compile Model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print('Loading Model...')
#model.load_weights('./models/first_model_action.h5')
#model.load_weights('./models/animal_asl_5_classes_1000_epoch_action.h5')
model.load_weights('./models/Epoch-144-Loss-0.53.h5')
print('Model Loaded!')

colors = [(245,221,173), (245,185,265), (146,235,193),(204,152,295),(255,217,179)]

def gen():
    # 1. New detection variables
    
    sequence = []
    sentence = []
    predictions = []
    #threshold = 0.5
    #threshold = min_detection_confidence
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    sent =''
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Read until video is completed
        
        while(cap.isOpened()):
        # Capture frame-by-frame

            global lstm_threshold
            #print('LSTM Model Detection Threshold = ',lstm_threshold)
            threshold = lstm_threshold

            global toggle_keypoints

            ret, image = cap.read()
            if ret == True:

                #print('threshold',threshold)

                # Make detections
                image, results = mediapipe_detection(image, holistic)
                #print(results)

                # Draw landmarks
                if toggle_keypoints:
                    draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                
                #print(keypoints)
                
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    #print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 

                            # Add overlay image if the most confident is more than the threshold
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
                    
                cv2.rectangle(image, (0,0), (700, 40), (0, 60, 123), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


                #encode output image to bytes
                frame = cv2.imencode('.jpg', image)[1].tobytes()

                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                #time.sleep(0.1)

            else: 
                break

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')





if __name__=="__main__":
    app.run(debug=True)