#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:46:58 2022

@author: ak
"""

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

clf = joblib.load("model.pkl")
initial_type = [('float_input',FloatTensorType([None,4]))]
onnx = convert_sklearn(clf,initial_types = initial_type)
with open("model.onnx","wb") as f:
    f.write(onnx.SerializeToString())

