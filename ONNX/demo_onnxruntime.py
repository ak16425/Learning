#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:13:07 2022

@author: ak
"""

#!wget https://github.com/onnx/models/blob/main/vision/classification/mnist/model/mnist-1.onnx


import json,sys,os,time,cv2,onnx,onnxruntime
from onnx import numpy_helper
import numpy as np

path = sys.argv[1]
#process the image
img = cv2.imread(path)
img = np.dot(img[...,:3],[0.299,0.587,0.114])
img = cv2.resize(img,dsize=(28,28),interpolation=cv2.INTER_AREA)
img.resize((1,1,28,28))

data = json.dumps({'data':img.tolist()})
data = np.array(json.loads(data)["data"]).astype("float32")
sess = onnxruntime.InferenceSession("mnist.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(input_name)
print(output_name)

result = sess.run([output_name],{input_name:data})
prediction = int(np.argmax(np.array(result).squeeze(),axis=0))
print(prediction)
