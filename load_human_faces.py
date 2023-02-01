"""
Randomly sample 2500 photos from UTKFace dataset of human faces deleteing all 
individuals younger than 10 and older than 80
"""

import shutil
import os
import random

dirpath = 'data/UTKFace'
destDirectory = 'data/human_face/'

filenames = random.sample(os.listdir(dirpath), 2500)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copy2(srcpath, destDirectory)