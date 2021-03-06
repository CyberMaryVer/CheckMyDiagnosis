from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.applications.mobilenet import preprocess_input
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
import cv2
import os

# returns a compiled model
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def png2rgb(png, background=(255,255,255) ):
    """Image converting in case if we get a link"""
    image_np = png
    row, col, ch = image_np.shape

    if ch == 3:
        return image_np

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2], image_np[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')

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
    # pred_prob = (pred_class).argsort().ravel()[::-1]
    pred_name_class = class_names[pred_class.argmax()].upper()
    pred_R = 100 - (100 * pred_class[:, 5][0])

    # create dictionary
    res = dict()
    for i, pr in enumerate(pred_class[0]):
        res.update({class_names[i].upper(): f'{100 * pr:.8f}%'})

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

    return (pred_name_class, pred_R, res)


app = Flask(__name__)
load_path = 'skin_model.h5'
# global model
model = load_model(load_path, custom_objects={"top_2_accuracy": top_2_accuracy, "top_3_accuracy": top_3_accuracy})
r = "test_image.jpg"

@app.route('/')
def home():
    return '<h1>working...</h1>'

@app.route('/test/', methods=['GET'])
def respond():
    # Retrieve the name from url parameter
    name = request.args.get("name", None)

    # For debugging
    print(f"Image url {name}")

    response = {}

    # Check if user sent an url
    if not name:
        response["ERROR"] = "no url found"
    # Valid url
    else:
        try:
            image_np = io.imread(name)
            img_inp = png2rgb(image_np)
            predictions = predict_one(img_inp, model)
            response.update({"Predicted type": predictions[0], "Predicted risk": predictions[1],
                             "Probabilities": predictions[2]})
        except Exception as ex:
            response["MESSAGE"] = f"Url {name}, {type(name)} is invalid"
            print(ex)

    # Return the response in json format
    return jsonify(response)


if __name__ == '__main__':

    port = os.environ.get('PORT')
    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()

    # >>>>>>>>>> DEBUG <<<<<<<<<<<<<<
    # img_url = 'test_image.jpg'
    # # get image and convert
    # # img_obj = url2rgb(img_url)
    #
    # img_obj = cv2.imread(img_url, cv2.COLOR_BGR2RGB) # random
    # # predict
    # predictions = predict_one(img_obj, model)
    # for pred in predictions:
    #     print(pred)




# if __name__ == '__main__':
#     # Threaded option to enable multiple instances for multiple user access support
#     app.run(threaded=True, port=5000)