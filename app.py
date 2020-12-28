from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
from PIL import Image
import albumentations as aug
from efficientnet_pytorch import EfficientNet

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model = torch.load("canc.pth")
model.eval()
def model_predict(file, model):
    image = Image.open(file)
    image = np.array(image)
    transforms = aug.Compose([
            aug.Resize(224,224),
            aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225),max_pixel_value=255.0,always_apply=True),
            ])
    image = transforms(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor([image], dtype=torch.float)
    preds = model(image)
    probs  = preds.detach().numpy()[0]
    probs = np.exp(probs)/np.sum(np.exp(probs))
    return preds,probs

@app.route('/')
def home():
    # Main page
    return render_template('home.html')


@app.route('/imageUpload',methods=['GET'])
def index():
    # image upload page
    return render_template('index.html')

@app.route('/covid',methods=['GET'])
def covid():
    # image upload page
    return render_template('covid.html')

@app.route('/bt',methods=['GET'])
def bt():
    # image upload page
    return render_template('bt.html')

@app.route('/malaria',methods=['GET'])
def malaria():
    # image upload page
    return render_template('malaria.html')

@app.route('/pneumonia',methods=['GET'])
def pneumonia():
    # image upload page
    return render_template('pneumonia.html')

@app.route('/alzymer',methods=['GET'])
def alzymer():
    # image upload page
    return render_template('alzymer.html')


@app.route('/profile',methods=['GET'])
def profile():
    # image upload page
    return render_template('profile.html')

@app.route('/predict', methods=['POST'])
def upload():
    # Get the file from post request
    f = request.files['file']
    labs=['MELANOMA (MALIGNANT)', 'MELANOCYTIC NEVUS (BENIGN)/ NORMAL SKIN /RASH', 'BASAL CELL CARCINOMA (BENIGN)', 'ACTINIC KERATOSIS (BENIGN)', 'BENIGN KERATOSIS (BENIGN)', 'DERMATOFIBROMA (NON CANCEROUS-BENIGN)', 'VASCULAR LESION (MAYBE BENIGN MAYBE MALIGNANT)', 'SQUAMOUS CELL CARCINOMA(MALIGNANT)']

    # Make prediction
    preds,probs = model_predict(f, model)
    result = labs[preds]
    prob_dict={}
    for x in range(len(labs)):
        prob_dict[labs[x]]=probs[x]
    return prob_dict


if __name__ == '__main__':
    app.run(debug=True)
