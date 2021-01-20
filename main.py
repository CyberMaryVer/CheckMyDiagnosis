from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.applications.mobilenet import preprocess_input
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import cv2

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/',methods=['POST'])
def predict_one(img, model, print_all=False, plot_img=False):
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

# returns a compiled model
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

if __name__ == '__main__':
    print('Main')
    load_path = 'skin_model.h5'
    model = load_model(load_path, custom_objects={"top_2_accuracy": top_2_accuracy, "top_3_accuracy": top_3_accuracy})
    print('Model is loaded', type(model))


