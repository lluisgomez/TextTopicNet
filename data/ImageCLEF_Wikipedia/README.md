
Download and uncompress here the **Wikipedia Retrieval 2010 Collection**.

To download the original dataset (~24 GB) follow the instructions provided here: http://imageclef.org/wikidata

A streamlined version of the dataset (~2.3 GB), with only JPEG images (resized to 256x256 pixels) and English articles, can be downloaded from https://goo.gl/jgQFGr

Our experiments make only use of the streamlined dataset, but the python scripts on the ``LDA/`` folder are prepared to process the entire original dataset and produce the streamlined version (see ``LDA/generate_train_dict.py``).

After downloading and uncompressing the dataset there must be three subdirectories here: 
``images/
  metadata/
  _text/
``
