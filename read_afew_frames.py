import numpy as np
from PIL import Image
import glob
import torch
import cv2


class ReadAFEWTrain:

    def __init__(self):
        self.add = ""
        self.filename = self.add + "AFEW2014/list_train.txt"
        self.base = self.add + "AFEW2014/Train/"
        self.labels_dict = {
            'Neutral': 0,
            'Disgust': 1,
            'Fear': 2,
            'Surprise': 3,
            'Happy': 4,
            'Sad': 5,
            'Angry': 6
        }

    def preprocessing(self, img, size=(256, 256)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, size).astype(np.float32)
        # img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        print("img size", img.shape)
        return img

    def get_dataset(self):
        with open(self.filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [x.split(" ") for x in content]
        train_videos = []
        train_labels = []
        for x in content:
            train_labels.append(self.labels_dict[x[1]])
            filelist = glob.glob(self.base + x[0] + '/*.jpg')
            video = np.array(
                [np.array(Image.open(fname)) for fname in filelist])
            #video=video/255.
            train_videos.append(video)
        return train_labels, train_videos


class ReadAFEWVal:

    def __init__(self):
        self.filename = "AFEW2014/list_eval.txt"
        self.base = "AFEW2014/Val/"
        self.add = ""
        self.filename = self.add + self.filename
        self.base = self.add + self.base
        self.labels_dict = {
            'Neutral': 0,
            'Disgust': 1,
            'Fear': 2,
            'Surprise': 3,
            'Happy': 4,
            'Sad': 5,
            'Angry': 6
        }

    def get_dataset(self):
        with open(self.filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [x.split(" ") for x in content]
        val_videos = []
        val_labels = []
        for x in content:
            val_labels.append(self.labels_dict[x[1]])
            filelist = glob.glob(self.base + x[0] + '/*.jpg')
            video = np.array(
                [np.array(Image.open(fname)) for fname in filelist])
            #video=video/255.
            #print(video)
            val_videos.append(video)
        #a=np.array(val_labels)
        #b=np.array(val_videos)
        #print(a.dtype)
        #print(b.dtype)
        return val_labels, val_videos
