from flask import Flask, render_template, request
# from scipy.misc import imsave,imread, imresize
import numpy as np
# import keras.models
import re
import base64
from PIL import Image
import sys 
import os
sys.path.append(os.path.abspath("./model"))
from model.model_arch import *
import torchvision.transforms as T
import torch
import cv2
import torch.nn as nn
from collections import OrderedDict

app = Flask(__name__)
global model, graph
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
global model
modelarch = LeNet5()
modelarch.load_state_dict(torch.load(r'lenet2.pth'))
transform = T.Compose(
    [T.ToTensor(),
    # torch.clamp(,min=0,max=1)
     T.Normalize([0.1307], [0.3081])])
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())
    #
    # modelarch = LeNet5()
    # modelarch.load_state_dict(torch.load(r'lenet.pth'))

    # read parsed image back in 8-bit, black and white mode (L)
    # x = Image.open('output.png').convert('L')
    x = cv2.imread('output.png',cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (28,28), interpolation=cv2.INTER_LINEAR)
    x1 = cv2.resize(x, (28, 28), interpolation=cv2.INTER_NEAREST)
    x2 = cv2.resize(x, (28, 28), interpolation=cv2.INTER_CUBIC)
    x = np.array(x)
    x = np.invert(x)
    # x = np.invert(x)


    # reshape image data for use in neural network
    x = transform(x)
    x = torch.unsqueeze(x,0)
    with torch.no_grad():
        images = x.to(device)
        outputs = modelarch(images).numpy()
        # print(outputs)
        outputs = np.argmax(outputs,1)
        # print(outputs)
        response = np.array_str(outputs)
        return response

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
