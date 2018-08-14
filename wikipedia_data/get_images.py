### git @ yash0307 ###

import logging
import os.path
import sys
import gensim
import bz2
import json
import sys,re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import utils, corpora, models
from gensim.corpora.wikicorpus import remove_markup, process_article, remove_template, filter_wiki
import logging
import re
from PIL import Image

if len(sys.argv) < 4:
	print('Please specify the following (in same order):')
	print('\t path to list of all images (list_of_images.json)')
	print('\t path of root directory where images are located')
	print('\t article number in wikipedia corpus for which images are required')
	print('An example of overall command is: python get_image.py /home/yash/list_of_images.json /media/DADES/yash/ 25')
	sys.exit(1)

list_of_images = json.load(open(sys.argv[1], 'r'))
root_dir = sys.argv[2]
text_article = sys.argv[3]

# Obtain directory of image paths with article number as key
def article_im_dir(list_of_images, root_dir):
	article_im_dir = {}
	for given_im in list_of_images:
		im_article_num = given_im.split('/')[2].split('#')[0]
		try:
			article_im_dir[im_article_num].append(root_dir+given_im)
		except KeyError:
			article_im_dir[im_article_num] = [root_dir+given_im]
	return article_im_dir

# Get a dictionary of all key = article number, value = image path
article_im_dir = article_im_dir(list_of_images, root_dir)

try:
	print(article_im_dir[text_article])
	for given_im in article_im_dir[text_article]:
		im = Image.open(given_im)
		print('Successfully loaded: ' + str(given_im))
except KeyError:
	print('No images found for given article number. This can be because the article number corresponds to a meta-file or article length is less than 50 words.')
