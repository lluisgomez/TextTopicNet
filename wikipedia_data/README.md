**TextTopicNet - Self-Supervised Learning of Visual Features Through Embedding Images on Semantic Text Spaces**

Y. Patel, L. Gomez, R. Gomez, M. Rusiñol, D. Karatzas, C.V. Jawahar.
```
@article{patel2018texttopicnet,
  title={TextTopicNet-Self-Supervised Learning of Visual Features Through Embedding Images on Semantic Text Spaces},
  author={Patel, Yash and Gomez, Lluis and Gomez, Raul and Rusi{\~n}ol, Mar{\c{c}}al and Karatzas, Dimosthenis and Jawahar, CV},
  journal={arXiv preprint arXiv:1807.02110},
  year={2018}
}
```
## Training Dataset

Download the image-text article co-occurring dataset from [here](http://datasets.cvc.uab.es/rrc/wikipedia_data/)
<br /> 
Make sure to download the following:
* All the images [here](http://datasets.cvc.uab.es/rrc/wikipedia_data/images/)
* Wikipedia text corpus dump [here](http://datasets.cvc.uab.es/rrc/wikipedia_data/text_dump/)
* List of images [here](http://datasets.cvc.uab.es/rrc/wikipedia_data/list_of_images.json) 

## Code Snippets
Following are some utility functions to access the data:
*The wikipedia text corpus dump associates each article with an unique id. To obtain the set of text documents used for training in our paper (text articles only with atleast 50 words), run the following command: ``python get_all_docs.py <path_to_list_of_all_images>``
* To obtain the list of images for a given text article number, run the following command: ``python get_images.py <path_to_list_of_all_images> <path_to_root_directory> <article_number>``

``get_images.py`` also provides a snippet to generate a python dictionary with key as text article number and value as list of co-occuring images.
