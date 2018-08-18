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

if len(sys.argv) < 2:
	print('Please specify the following (in same order):')
	print('\t path to list of all images (list_of_images.json)')
	print('An example of overall command is: python get_image.py /home/yash/list_of_images.json')
	sys.exit(1)

list_of_images = json.load(open(sys.argv[1], 'r'))

# Obtain directory of image paths with article number as key
def all_articles(list_of_images):
	list_articles = []
	for given_im in list_of_images:
		im_article_num = given_im.split('/')[2].split('#')[0]
		list_articles.append(im_article_num)
	return list_articles

# Get a list of all articles used in training
list_articles = list(set(all_articles(list_of_images)))

# Dump the list of all articles
json.dump(list_articles, open('list_all_articles.json', 'w'))
