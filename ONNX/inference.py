#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:58:40 2022

@author: ak
"""

import onnxruntime as rt
import numpy as np
data = np.array([[6.7, 3.3, 5.7, 2.5],
       [6.7, 3. , 5.2, 2.3],
       [6.3, 2.5, 5. , 1.9],
       [.5, 1. , 1.2, 2. ]])
sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onnx = sess.run([label_name],{input_name: data.astype(np.float32)})[0]
print(pred_onnx)