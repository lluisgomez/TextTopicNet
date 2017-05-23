import caffe
import numpy as np
from PIL import Image
import json
import random
import time


class MultiLabelDataLayerWikiCLEF(caffe.Layer):
    """
    Load (input image, label image) pairs from ImageCLEF_Wikipedia
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - img_dir: path to ImageCLEF_Wikipedia images dir
        - mean: tuple of mean values to subtract
        - num_topics: dimensionality of LDA topic space, i.e. last fc layer dim
        - batch_size: ...

        example

        params = dict(img_dir="/media/DADES/datasets//images/",
            mean=(104.00698793, 116.66876762, 122.67891434)
            batch_size=64, num_topics=40)
        """
        # config
        params = eval(self.param_str)
        self.img_dir = params['img_dir']
        self.mean = np.array(params['mean'])
        self.batch_size = params['batch_size']
        num_topics = params['num_topics']

        # input data placeholders
        self.data   = np.zeros((self.batch_size,3,227,227), dtype=np.float32)
        self.label  = np.zeros((self.batch_size,1,num_topics,1), dtype=np.float32)

        # load GT from json
        f = open('../../LDA/training_labels'+str(num_topics)+'.json');
        self.gt_train_dict = json.load(f);
        f.close();

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # indices for images and labels
        self.idx = 0

    def reshape(self, bottom, top):
        # load image + label image pair
        self.load_images(self.gt_train_dict.keys()[self.idx:self.idx+self.batch_size])
        self.load_labels(self.gt_train_dict.values()[self.idx:self.idx+self.batch_size])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        self.idx += self.batch_size
        if self.idx+self.batch_size > len(self.gt_train_dict):
          self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_images(self, idxs):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        for i,idx in enumerate(idxs):
          im = Image.open(self.img_dir+idx)
          im = im.resize((256,256)) # resize to 256x256
          # data augmentation by random crops
          offset_x = random.randint(0,29)
          offset_y = random.randint(0,29)
          im = im.crop((offset_x,offset_y,227+offset_x,227+offset_y)) # crop of 227x227
          if random.randint(0,1) == 1:
            im = im.transpose(Image.FLIP_LEFT_RIGHT) # data augmentation by random mirror
          if len(np.array(im).shape) < 3:
            im = im.convert('RGB') # grayscale to RGB
          in_ = np.array(im, dtype=np.float32)
          in_ = in_[:,:,::-1] # switch channels RGB -> BGR
          in_ -= self.mean # subtract mean
          in_ = in_.transpose((2,0,1)) # transpose to channel x height x width order
          self.data[i,:,:,:] = in_

    def load_labels(self, idxs):
        """
        """
        self.label = np.array(idxs, dtype=np.float32)
