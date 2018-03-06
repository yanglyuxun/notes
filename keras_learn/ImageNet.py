#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
use the model of 
ImageNet Large Scale Recognition Challenge (ILSVRC)

available models:
VGG16
VGG19
ResNet50
Inception v3
CRNN for music tagging

url:
https://github.com/fchollet/deep-learning-models

model dir:
~/.keras/models/
"""
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras import applications
import numpy as np
from requests import Session
from PIL import Image as pil_image
import pandas as pd
import os

def get_model(name='VGG19'):
    if name=='VGG16':
        return applications.VGG16(),(224, 224)
    elif name=='VGG19':
        return applications.VGG19(),(224, 224)
    elif name=='ResNet50':
        return applications.ResNet50(),(224, 224)
    elif name=='InceptionV3':
        return applications.InceptionV3(),(299,299)
    elif name=='MobileNet':
        return applications.MobileNet(),(224, 224) # any>32*32
    elif name=='Xception':
        return applications.Xception(),(299,299)
    else:
        return None, None
class classifier(object):
    def __init__(self,model_name=['VGG19'],show_net=False):
        self.model_name = model_name
        self.model, self.size = {},{}
        for name in model_name:
            self.model[name], self.size[name] = get_model(name)
            if self.model[name] is None:
                print('Error: no such model:"%s".'%name)
                self.model.pop(name)
                self.size.pop(name)
            else:
                if show_net: 
                    print('The structure of %s:'% name)
                    print(self.model.summary())
        self.sess = Session()
    def _img_class(self,img):
        preds = {}
        for name in self.model_name:
            print('Using',name,'...')
            if self.size:
                img = img.resize(self.size[name])
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds[name] = self.model[name].predict(x)
        return preds
    def _pred2df(self, preds, top_results,prob_th):
        dfs = {}
        for name in self.model_name:
            decoded = decode_predictions(preds[name], top=1000)[0]
            if decoded[top_results][2]<prob_th:
                decoded = decoded[:top_results]
            else:
                decoded = [d for d in decoded if d[2]>=prob_th]
            dfs[name] = pd.DataFrame(decoded,
                                columns=('Class','Name','Prob'))
        return dfs
    def class_path(self,img_path='test.jpeg', 
             top_results=5,
             prob_th=0.05):
        img = image.load_img(img_path)
        preds = self._img_class(img)
        return self._pred2df(preds,top_results,prob_th)
    def class_url(self,url,top_results=5,
                  prob_th=0.05):
        img_b = self.sess.get(url).content
        img = pil_image.open(fake_file(img_b))
        preds = self._img_class(img)
        return self._pred2df(preds,top_results,prob_th)
    def clean_cache(self,dir='~/.keras/models/'):
        dir=os.path.expanduser(dir)
        ls = os.listdir(dir)
        for f in ls:
            if f.endswith('.h5'):
                os.remove(dir+f)
                print('Removed:',f)
        
class fake_file(object):
    def __init__(self, read_data):
        self.d=read_data
    def read(self):
        return self.d

all_model = ['VGG16','VGG19','ResNet50','InceptionV3','MobileNet','Xception']
c = classifier(all_model)
c.class_path()
image_url='https://orig00.deviantart.net/45aa/f/2015/323/9/9/letter_a_by_hillygon-d9h8c6a.jpg'
c.class_url(image_url)
#c = classifier('VGG16')
#c.class_path()
#c.class_url(image_url)
#c = classifier('VGG19')
#c.class_path()
#c.class_url(image_url)
#c = classifier('ResNet50')
#c.class_path()
#c.class_url(image_url)
#c = classifier('InceptionV3')
#c.class_path()
#c.class_url(image_url)
#c = classifier('MobileNet')
#c.class_path()
#c.class_url(image_url)
#c = classifier('Xception')
#c.class_path()
#df = c.class_url(image_url)
#df.to_html()
if __name__=='__main__':
    pass
