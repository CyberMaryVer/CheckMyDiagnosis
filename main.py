from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.applications.mobilenet import preprocess_input
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import tensorflow as tf
import json
import cv2

# returns a compiled model
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def url2rgb(url, background=(255,255,255) ):
    """Image converting in case if we get a link"""
    image_np = io.imread(url)
    row, col, ch = image_np.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')

def predict_one(img, model=model, print_all=False, plot_img=False):
    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    preprocessed = preprocess_input(resized)
    input_img = preprocessed.reshape(1, 224, 224, 3)

    class_names = {
        0: 'akiec',  # actinic keratoses and intraepithelial carcinoma/Bowen disease (akiec)
        1: 'bcc',  # basal cell carcinoma (bcc) *
        2: 'bkl', # benign lesions of the keratosis type
        3: 'df',  # dermatofibroma (df)
        4: 'mel',  # melanoma (mel) *
        5: 'nv',  # melanocytic nevi (nv)
        6: 'vasc'  # vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhages, vasc)
    }
    pred_class = model.predict(input_img)
    pred_prob = (pred_class).argsort().ravel()[::-1]
    pred_name_class = class_names[pred_class.argmax()].upper()
    pred_R = 100 - (100 * pred_class[:, 5][0])

    # print in red if the risk of melanoma is high
    if pred_R > 5:
        f1, f2 = '\x1b[31m', '\x1b[0m'
    else:
        f1, f2 = '', ''

    print(f'Predicted class: {pred_name_class}, predicted risk: {f1}{pred_R:.8f}{f2}%')

    if print_all:
        print()
        for i, pr in enumerate(pred_class[0]):
            print(f'Probability of type {class_names[i].upper()}: {100 * pr:.8f}%')

    if plot_img:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        plt.imshow(img_gray, cmap='gray')
        plt.axis('off')
        plt.title('Mole')
        plt.show()

    return (pred_name_class, pred_class, pred_R)


app = Flask(__name__)
load_path = 'skin_model.h5'
global model
model = load_model(load_path, custom_objects={"top_2_accuracy": top_2_accuracy, "top_3_accuracy": top_3_accuracy})
r = "test_image.jpg"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/',methods=['POST'])
def predict(): ################## pseudo-code
    data = request.get_json() ###################################
    print(data) ################################################
    img_url = json.load(data)['imgurl']
    img_obj = url2rgb(img_url)
    predictions = predict_one(img_obj, model)

    pass

if __name__ == '__main__':
    print('Main')
    print('Model is loaded', type(model))
    app.run(debug=True, host='0.0.0.0')


