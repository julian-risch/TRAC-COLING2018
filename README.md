# TRAC-COLING2018

The First Shared Task on Aggression Identification dealt with the classification of the aggression level of user posts at different social media platforms. It was part of the Workshop on Trolling, Aggression and Cyberbullying (TRAC 2018) at the International Conference of Computational Linguistics (COLING 2018).

### Citation
If you use our work, please cite our paper [**Aggression Identification Using Deep Learning and Data Augmentation**](https://github.com/julian-risch/TRAC-COLING2018/raw/master/risch2018aggression.pdf) as follows:

    @inproceedings{risch2018aggression,
    author = {Risch, Julian and Krestel, Ralf},
    booktitle = {Proceedings of the Workshop on Trolling, Aggression and Cyberbullying (TRAC@COLING)},
    title = {Aggression Identification Using Deep Learning and Data Augmentation},
    pages = {150--158},
    year = {2018}
    }


### Implementation
This directory contains the implementation of our submission to the shared task.
* `coling-oof.py` generates out-of-fold predictions for four different approaches.
* `coling-lgbm` combines these four approaches into one ensemble and generates final predictions for the tasks test data.

### Dataset
We publish the augmented dataset under the Creative Commons Non-Commercial Share-Alike 4.0 licence [**CC-BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

The original dataset (`agr_en_train.csv`, `agr_en_dev.csv`) has been provided by the organisers of the First Shared Task on Aggression Identification: https://sites.google.com/view/trac1/shared-task Contact: coling.aggression[at]gmail[dot]com

The other six files (`train_de.csv`, `train_es.csv`, `train_fr.csv`, `dev_de.csv`, `dev_es.csv`, `dev_fr.csv`) have been created by us (please see the paper for details).

`agr_en_train.csv` contains the training dataset for the shared task in English. 
`agr_en_dev.csv` contains the validation (or development) dataset for the shared task in English.

Both the files contain 3 columns in the following format
		`unique_id,text,aggression-level`

The columns are separated by comma and follows a minimal quoting pattern (such that only those columns are quoted which are in multiple lines or contain quotes in the text).

`train_de.csv`, `train_es.csv`, and `train_fr.csv` contain translations of the train set for the shared task in English. More specifically, the train set has been machine translated into German and afterwards translated back into English to generate `train_de.csv`. Similarly, the train set has been machine translated into Spanish and French and afterwards back into English. This has been done to augment the original training set and to increase the number of training examples. The machine translation adds variety to the training examples.
