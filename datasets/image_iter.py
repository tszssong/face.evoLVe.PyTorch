import os, sys, time
import numpy as np
import cv2
import collections
import torch
import torch.utils.data as data

class FaceImageIter(data.Dataset):
    def __init__(self, 

