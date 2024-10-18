from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading

app = Flask(__name__)

label = "Warmup...."
n_time_steps = 10
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Tải mô hình
model = tf.keras.models.load_model('C:\\Users\\hoang\\Downloads\\NDDT\\Hao\\Human_Activity_Recognition-main\\my_model.keras')

# Khởi tạo VideoCapture (sử dụng camera mặc định)
cap = cv2.VideoCapture(0)

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    if results[0][0] > 0.5:
        label = "RUNNING"
    elif results[0][1] > 0.5:
        label = "SITTING"
    elif results[0][2] > 0.5:
        label = "WALKING"
    elif results[0][3] > 0.5:
        label = "STANDING"
    elif results[0][4] > 0.5:
        label = "HANDSWING"
    else:
        label = "UNKNOWN"
    return label

def generate_frames():
    i = 0
    warmup_frames = 60

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            i += 1
            if i > warmup_frames:
                if results.pose_landmarks:
                    c_lm = make_landmark_timestep(results)
                    lm_list.append(c_lm)
                    if len(lm_list) == n_time_steps:
                        # Dự đoán
                        t1 = threading.Thread(target=detect, args=(model, lm_list,))
                        t1.start()
                        lm_list.clear()
                    img = draw_landmark_on_image(mpDraw, results, img)

            img = draw_class_on_image(label, img)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
