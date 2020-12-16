from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
from skimage.transform import resize

# Flask utils
from flask_ngrok import run_with_ngrok
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
run_with_ngrok(app)  

# Model saved with Keras model.save()
MODEL_PATH = '/Webapp_Colorize/models/baseline_mse_finetune_05-0.011.h5'

# Load your trained model
model = load_model('/Webapp_Colorize/models/baseline_mse_finetune_05-0.011.h5')
model.load_weights('/Webapp_Colorize/models/baseline_mse_finetune_05-0.011.h5')
#model._make_predict_function()          # Necessary
i = 0
#print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
#     img_width, img_height = 150, 150
#     x = image.load_img(img_path, target_size=(img_width, img_height))
#     x = image.img_to_array(x)
#     x = np.expand_dims(x, axis=0)
#     pred = model.predict(x)
#     return pred

def to_rgb(gray, ab):
    '''rgb image from grayscale and ab channels'''
    ab = ab*128
    lab_img = np.concatenate((gray,ab),axis=2)
    rgb_img = lab2rgb(lab_img)
    # rgb_img = (rgb_img*255).astype('uint8')
    return rgb_img

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/colorize', methods=['GET', 'POST'])
def upload():
    global i
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname('/Webapp_Colorize/')
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        test_images = image.img_to_array(image.load_img(file_path))
        test_images = resize(test_images,(256,256))
        img1_color = []
        img1_color.append(test_images)
        img1_color = np.array(img1_color, dtype=float)
        img1_color = rgb2lab(1.0/255 * img1_color)[:,:,:,0]
        img1_color = img1_color.reshape(img1_color.shape+(1,))

        output1 = model.predict(img1_color)
        output1 *= 128

        result = np.zeros((256,256,3))
        result[:,:,0] = img1_color[0][:,:,0]
        result[:,:,1:] = output1[0]
        output = lab2rgb(result)
        #plt.imshow(output)
        i += 1
        plt.imsave("/Webapp_Colorize/static/colorized"+str(i)+".png",output)
        return "colorized"+str(i)+".png"
    return None

app.run()
