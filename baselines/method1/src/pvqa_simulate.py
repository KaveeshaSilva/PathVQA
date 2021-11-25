# coding=utf-8

import base64
import json
import os
import pandas as pd
import pickle
import cv2

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

import re
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu

from tasks.pvqa_model import PVQAModel

baseUrl = 'drive/MyDrive/PathVQA/'
model_dir = baseUrl+"checkpointtemp_LXRT"


FIELDNAMES = ['image_id', 'image_w', 'image_h',
              'num_boxes', 'boxes', 'features']


def load_tsv(split: str):
    tsv_file = baseUrl+'data/pvqa/images/%s%s.csv' % (split, args.pvqaimgv)
    df = pd.read_csv(tsv_file, delimiter='\t', names=FIELDNAMES)

    data = []
    for i in range(df.shape[0]):
        datum = {}
        datum['img_id'] = '%s_%04d' % (split, df['image_id'][i])
        datum['img_w'] = df['image_w'][i]
        datum['img_h'] = df['image_h'][i]
        datum['num_boxes'] = df['num_boxes'][i]

        boxes = df['boxes'][i]
        buf = base64.b64decode(boxes[1:])
        temp = np.frombuffer(buf, dtype=np.float64).astype(np.float32)
        datum['boxes'] = temp.reshape(datum['num_boxes'], -1)

        features = df['features'][i]
        buf = base64.b64decode(features[1:])
        temp = np.frombuffer(buf, dtype=np.float32)
        datum['features'] = temp.reshape(datum['num_boxes'], -1)

        data.append(datum)

    return data


if __name__ == '__main__':

    splits = ['test']
    # loading detection features to img_data
    imgid2img = {}
    for split in splits:
        data = load_tsv(split)

        for datum in data:
            imgid2img[datum['img_id']] = datum

    label2ans = pickle.load(
        open(baseUrl+'data/pvqa/qas/trainval_label2ans.pkl', 'rb'))

    model = PVQAModel(4092)
    state_dict = torch.load("%s.pth" % model_dir)
    model.load_state_dict(state_dict["model_state_dict"])
    model = model.cuda()

    while True:
        cv2.destroyAllWindows()
        filePath = '/gdrive/My Drive/PathVQA/pvqa/images/test/'
        print('Enter image name (ex:- test_0001):')
        img_id = input().strip()
        imgnameParts=img_id.split("_")
        if(imgnameParts[0]!="test"):
            if(len(imgnameParts[1])!=4):
                print('Wrong image name..')

        filePath+=img_id+".jpg"
        image = cv2.imread(filePath)
        cv2.imshow(img_id,image)

        image_fet = imgid2img[img_id]
        feats = torch.tensor([image_fet['features']])
        boxes = torch.tensor([image_fet['boxes']])

        x = 'y'

        while (x in ['y', 'Y']):
            print('Enter a question:')
            q = input().strip()

            # predict and print max 5
            with torch.no_grad():
                
                feats, boxes = feats.cuda(), boxes.cuda()

                sents = [q]
                targets = ['yes']

                logit = model(feats, boxes, sents, targets)
                score, label = logit.max(1)

            print(label2ans[label])

            print('Question about previous image? (y/n)')
            x = input().strip()
