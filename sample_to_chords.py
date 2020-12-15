# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:27:20 2020

@author: louis
"""
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import data_loader
import numpy as np

PITCH_LIST = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
QUALITY_LIST = ["maj", "min", "dim", "maj7", "min7", "7", "dim7"]


def sample_to_chords(sample):
    
    idx_chords = np.argmax(sample[0,:,:],1)

    chords = [PITCH_LIST[int(idx%12)]+":"+QUALITY_LIST[int(idx/12)] for idx in idx_chords]
    
    return chords
