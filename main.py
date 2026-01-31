from flask import Flask, request, render_template, flash
import tensorflow as tf

import os
import numpy as np
from werkzeug.utils import secure_filename

try:
    import cv2
except Exception:
    cv2 = None

app = Flask(__name__)

# CTCLoss Definition

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Loading Model

with tf.keras.utils.custom_object_scope({'CTCLoss': CTCLoss}):
    model = tf.keras.models.load_model('models/model.keras', compile=False)

# Defining vocab for string generation

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


# Data Loader

def load_data(path:str):
    path = bytes.decode(path.numpy())
    frames = np.load(path)
    frames = tf.cast(frames,tf.float32)
    return frames

def mappable_function(path: str):
    result = tf.py_function(load_data, [path], tf.float32)
    return result


# Simple preprocessing for uploaded videos: extract or sample 75 frames,
# convert to grayscale, resize to (70x50), normalize and save as .npy
def preprocess_uploaded_video(video_filepath, out_npy_path, target_frames=75, width=70, height=50):
    if cv2 is None:
        raise RuntimeError('OpenCV (cv2) is required for preprocessing uploaded videos. Install opencv-python.')

    cap = cv2.VideoCapture(video_filepath)
    frames = []
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        # convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (width, height))
        frames.append(resized.astype(np.float32) / 255.0)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError('Uploaded video contains no frames.')

    # sample or pad to target_frames
    if len(frames) >= target_frames:
        idxs = np.linspace(0, len(frames) - 1, target_frames).astype(int)
        sampled = [frames[i] for i in idxs]
    else:
        sampled = frames[:] + [np.zeros((height, width), dtype=np.float32) for _ in range(target_frames - len(frames))]

    arr = np.stack(sampled, axis=0)  # shape (target_frames, height, width)
    arr = np.expand_dims(arr, axis=-1)  # add channel dim -> (target_frames, height, width, 1)
    np.save(out_npy_path, arr)
    return out_npy_path

# Predicting videos

def model_predict(video_path):
    data = tf.data.Dataset.list_files(video_path)
    data = data.map(mappable_function)
    data = data.padded_batch(1, padded_shapes=[75,50,70,1])
    
    yhat = model.predict(data)
    decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=False)[0][0].numpy()
    predicted_text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode('utf-8')


    return predicted_text

# Landing page route

@app.route('/', methods=['GET'])
@app.route('/index.html')
def index():
    return render_template('index.html')

# Prediction route

@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    # If a file was uploaded, handle it; otherwise use the selected preset
    uploaded = request.files.get('videofile')
    selected = request.form.get('videoselect')

    if uploaded and uploaded.filename != '':
        filename = secure_filename(uploaded.filename)
        name, ext = os.path.splitext(filename)
        save_video_dir = os.path.join('static', 'media', 'video')
        save_backend_dir = os.path.join('static', 'media', 'backend')
        os.makedirs(save_video_dir, exist_ok=True)
        os.makedirs(save_backend_dir, exist_ok=True)

        saved_video_path = os.path.join(save_video_dir, filename)
        uploaded.save(saved_video_path)

        numpy_fname = f"{name}.npy"
        numpy_path = os.path.join(save_backend_dir, numpy_fname)

        try:
            preprocess_uploaded_video(saved_video_path, numpy_path)
        except Exception as e:
            return render_template('result.html', video_path=saved_video_path, result=f'Preprocessing failed: {e}', original='')

        txt_path = os.path.join(save_backend_dir, f"{name}.txt")
        # create placeholder txt so code reading original won't fail
        if not os.path.exists(txt_path):
            with open(txt_path, 'w') as f:
                f.write('Uploaded video')

        video_path = saved_video_path.replace('\\', '/')
        predicted = model_predict(numpy_path)
        with open(txt_path, 'r') as f:
            original = f.readline()
        return render_template('result.html', video_path=video_path, result=predicted, original=original)

    # fallback to selected preset
    if not selected:
        return render_template('result.html', video_path='', result='No video selected', original='')

    video = selected
    numpy_path = os.path.join('static', 'media', 'backend', f'{video}.npy')
    txt_path = os.path.join('static', 'media', 'backend', f'{video}.txt')
    video_path = f'static/media/video/{video}.mp4'

    predicted = model_predict(numpy_path)

    with open(txt_path, 'r') as f:
        original = f.readline()

    return render_template('result.html', video_path=video_path, result=predicted, original=original)

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)