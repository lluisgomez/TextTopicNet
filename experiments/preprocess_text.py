import sys
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import utils, corpora, models
from gensim.corpora.wikicorpus import filter_wiki, remove_file

filter_more = re.compile('(({\|)|(\|-)|(\|})|(\|)|(\!))(\s*\w+=((\".*?\")|([^ \t\n\r\f\v\|]+))\s*)+(({\|)|(\|-)|(\|})|(\|))?', re.UNICODE | re.DOTALL | re.MULTILINE) 
def preprocess_imageclef(raw):
    # Initialize Tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # Initialize Lemmatizer
    lemma = WordNetLemmatizer()
    
    # create English stop words list
    en_stop = get_stop_words('en')
    
    # Decode Wiki Markup entities and remove markup
    text = filter_wiki(raw)
    text = re.sub(filter_more, '', text)

    # clean and tokenize document string
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    
    # remove stop words from tokens
    tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    tokens = [lemma.lemmatize(i) for i in tokens]

    # remove non alphabetic characters
    tokens = [re.sub(r'[^a-z]', '', i) for i in tokens]
    
    # remove unigrams and bigrams
    tokens = [i for i in tokens if len(i)>2]
    
    return (tokens, text)

def preprocess_wikidata(raw):
 # Initialize Tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # Initialize Lemmatizer
    lemma = WordNetLemmatizer()

    # create English stop words list
    en_stop = get_stop_words('en')

    # Decode Wiki Markup entities and remove markup
    text = filter_wiki(raw)
    text = re.sub(filter_more, '', text)

    # clean and tokenize document string
    text = text.lower().split('../img/')[0]
    tokens = tokenizer.tokenize(text)

    # remove stop words from tokens
    tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    tokens = [lemma.lemmatize(i) for i in tokens]

    # remove non alphabetic characters
    tokens = [re.sub(r'[^a-z]', '', i) for i in tokens]

    # remove unigrams and bigrams
    tokens = [i for i in tokens if len(i)>2]

    return (tokens, text)

