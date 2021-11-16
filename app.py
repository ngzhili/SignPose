from mediapipe_functions import mediapipe_detection, draw_landmarks, draw_styled_landmarks, extract_keypoints, add_image, prob_viz, overlay_transparent
from flask import Flask, render_template, Response, request, json, jsonify
from flask_socketio import SocketIO

import cv2
import time
import sys

import random
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
#from tensorflow.keras.callbacks import TensorBoard

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

sys.path.append('./mediapipe_functions.py')

video_id = 'no'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Initialise detection confidence
lstm_threshold = 0.5
toggle_keypoints = True
mediapipe_detection_confidence = 0.5


@app.route('/process_toggle_value', methods=['POST', 'GET'])
def process_toggle_value():
    global toggle_keypoints
    if request.method == "POST":
        toggle_data = request.get_json()
        # print(toggle_data)
        # print(slider_data[0]['slider'])
        toggle_keypoints = toggle_data[0]['toggle']
        print('Toggle_keypoints:', toggle_keypoints)
    results = {'processed': 'true'}
    return jsonify(results)


@app.route('/process_slider_value', methods=['POST', 'GET'])
def process_slider_value():
    global lstm_threshold

    if request.method == "POST":
        slider_data = request.get_json()
        # print(slider_data)
        # print(slider_data[0]['slider'])
        lstm_threshold = float(slider_data[0]['slider'])
        print('LSTM Detection Threshold:', lstm_threshold)
    results = {'processed': 'true'}
    return jsonify(results)


def random_action():
    global current_action
    newAction = random.choice(actions_list)

    # while the new action is equal to the previous action, choose a new action
    while newAction == current_action:
        newAction = random.choice(actions_list)

    current_action = newAction
    #print('Current Action:', current_action)
    return current_action


@socketio.on('generate new action')
def emit_new_action():
    random_action()
    socketio.emit('new action', {'data': current_action})


@app.route("/get_current_action", methods=['GET'])
def get_current_action():
    return jsonify(current_action)


@app.route("/get_next_action", methods=['GET'])
def get_next_action():
    current_action = random_action()
    return jsonify(current_action)


@app.route("/get_current_score", methods=['GET'])
def get_current_score():
    return jsonify(current_score)

reset_score_frame_count = 0

@app.route('/reset_score')
def reset_score():
    global current_score
    global reset_score_frame_count
    global sentence
    current_score = 0
    sentence = []

    reset_score_frame_count = 10
    print('current_score', current_score)

    return("nothing")


''' ============== Build Mediapipe Model ============== '''
# Use Holistic Models for detections
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Actions that we try to detect
#actions = np.array(['hello', 'thanks', 'iloveyou'])
#actions = np.array(['Alligator','Butterfly','Cow','Elephant','Gorilla'])

actions = np.array(['Bird', 'Butterfly', 'Cow',
                   'Elephant', 'Gorilla', 'No Action'])
actions_list = list(['Bird', 'Butterfly', 'Cow',
                     'Elephant', 'Gorilla'])
current_action = random.choice(actions_list)


@app.route('/', methods=['GET', 'POST'])
def index():
    # if request.method == 'POST':
    # if request.form.get('LSTM_model') == 'start_detection':
    # print('hello123')
    #video_id = 'lstm'
    """Video streaming home page."""
    return render_template('index.html', current_action=current_action, current_score=current_score)


label_map = {label: num for num, label in enumerate(
    actions)}  # create label map dictionary

''' ============== Build LSTM Model using Keras ============== '''
#model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
#model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
#model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(actions.shape[0], activation='softmax'))


# Build Model Architecture (Body pose and Handpose only)
model = Sequential()
# each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(64, return_sequences=True,
          activation='relu', input_shape=(30, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
# next layer is a dense layer so we do not return sequences here
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile Model
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

print('Loading Model...')

''' ============== Load Model Weights ============== '''
# model.load_weights('./models/first_model_action.h5')
# model.load_weights('./models/animal_asl_5_classes_1000_epoch_action.h5')
# model.load_weights('./models/Epoch-144-Loss-0.53.h5')
# model.load_weights('./models/animal_6_classes_Epoch-237-Loss-0.12.h5')
model.load_weights('./models/run6.h5')

print('Model Loaded!')

colors = [(245, 221, 173), (245, 185, 265), (146, 235, 193),
          (204, 152, 295), (255, 217, 179), (0, 0, 179)]

current_score = 0

''' ============== Mediapipe & LSTM Detection Code ============== '''

sentence = []


def gen():
    # 1. New detection variables

    sequence = []
    #sentence = []
    global sentence
    predictions = []
    #threshold = 0.5
    #threshold = min_detection_confidence
    """Video streaming generator function."""
    #cap = cv2.VideoCapture(0)


    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0

    #print('Current Action:', current_action)

    global current_score
    global reset_score_frame_count
    global mediapipe_detection_confidence
    # print(mediapipe_detection_confidence)

    print('gen started')

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Read until video is completed

        while(cap.isOpened()):
            # Capture frame-by-frame

            global lstm_threshold
            #print('LSTM Model Detection Threshold = ',lstm_threshold)
            threshold = lstm_threshold

            global toggle_keypoints

            ret, image = cap.read()

            width = image.shape[1]  # 480
            height = image.shape[0]  # 640
            #print(width,height)

            if ret == True:
                annotated_image = image.copy()

                # Make detections
                image, results = mediapipe_detection(image, holistic)


                ''' ===== Prediction logic ===== '''
                keypoints = extract_keypoints(results)


                # append the most recent 30 frames of keypoints
                sequence.append(keypoints)
                sequence = sequence[-30:]


                ''' ====== loading model screen ====== '''
                if len(sequence) < 30:
                    width = image.shape[1]  # 480
                    height = image.shape[0]  # 640
                    alpha = 0.5

                    overlay = image.copy()

                    cv2.rectangle(overlay, (0, 0), (width, height),
                                  (255, 255, 255), -1)

                    # apply the overlay
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha,
                                    0, image)
                    
                    (text_width, text_height), baseline = cv2.getTextSize('Loading...', cv2.FONT_HERSHEY_SIMPLEX,1, 2)


                    cv2.putText(image, 'Loading...', (width//2 - text_width//2, height//2 + text_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                
                ''' ====== draw landmarks ====== '''
                # Draw landmarks when lstm model is ready
                if toggle_keypoints and len(sequence) == 30:
                    draw_styled_landmarks(image, results)



                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    # print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    #print(predictions[-10:])
                    #print(np.unique(predictions[-10:]))
                    #print(np.unique(predictions[-10:])[0])
                    #print(np.unique(predictions[-10:]))
                    #print(np.full((1,10),np.argmax(res)))

                    ''' ===== Vizualization logic ===== '''

                    # the last class index prediction is equal to the highest predicted class index in result
                    if np.unique(predictions[-10:])[0] == np.argmax(res): 
                        
                        # if green screen is no longer displayed, then checks if action is correct
                        ''' ===== Checks if Action is Correct ===== '''
                        if res[np.argmax(res)] >= threshold and actions[np.argmax(res)] == current_action and frame_count == 0:

                            print('Correct!')
                            #current_action = random.choice(actions_list)
                            frame_count = 15  # 15

                            emit_new_action()
                            #print('Current Action:', current_action)

                            current_score += 1

                            if len(sentence) > 0:
                                # if action is not in the last sentence, then we append the last action to the sentence
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        # display most confidence animal emoji
                        if res[np.argmax(res)] >= threshold and actions[np.argmax(res)] != 'No Action':
                            # print(actions[np.argmax(res)])

                            # Add overlay image if the most confident is more than the threshold
                            add_image(image, results, str(
                                actions[np.argmax(res)]))

                    if len(sentence) > 15:
                        sentence = sentence[-15:]

                    ''' ====== Class Probability Display ===== '''
                    image = prob_viz(res, actions, image, colors, threshold)

                ''' ====== Correct Classes Top Display Bar ===== '''

                cv2.rectangle(image, (0, 0), (width, 50), (0, 60, 123), -1)
                
                if len(sentence) > 0:
                    cv2.putText(image, ' ' + sentence[-1], 
                        (3, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 221, 173), 2, cv2.LINE_AA)
                    
                    (text_width, text_height), baseline = cv2.getTextSize(sentence[-1], cv2.FONT_HERSHEY_SIMPLEX,1, 2)
                    
                if len(sentence)> 1:
                    
                    cv2.putText(image, '  ' + ' '.join(
                        sentence[::-1][1:]), (text_width, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



                ''' ====== display correct screen ====== '''
                # display_correct_screen if frame_count is more than 0
                if frame_count > 0:
                    # display_correct_screen(image)
                    width = image.shape[1]  # 480
                    height = image.shape[0]  # 640
                    alpha = 0.5

                    overlay = image.copy()

                    cv2.rectangle(overlay, (0, 0), (width, height),
                                  (144, 250, 144), -1)

                    # apply the overlay
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha,
                                    0, image)

                    cv2.putText(image, 'CORRECT!', (width//2 - 75, height//2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    overlay = cv2.imread(
                        './static/images/correct1.png', cv2.IMREAD_UNCHANGED)
                    image = overlay_transparent(
                        image, overlay, width//2 - 35, height//2-70)

                    frame_count -= 1

                ''' ====== display reset score screen ====== '''
                if reset_score_frame_count > 0:
                    width = image.shape[1]  # 480
                    height = image.shape[0]  # 640
                    alpha = 0.5

                    overlay = image.copy()

                    cv2.rectangle(overlay, (0, 0), (width, height),
                                  (255, 255, 255), -1)

                    # apply the overlay
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha,
                                    0, image)
                    
                    (text_width, text_height), baseline = cv2.getTextSize('Score Reset!', cv2.FONT_HERSHEY_SIMPLEX,1, 2)

                    cv2.putText(image, 'Score Reset!', (width//2 - text_width//2, height//2 + text_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    reset_score_frame_count -= 1


                ''' ===== display screen to html ===== '''
                # encode output image to bytes
                frame = cv2.imencode('.jpg', image)[1].tobytes()

                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                # time.sleep(0.1)

            else:
                break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    print('video feed running')

    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socketio.run(app)
