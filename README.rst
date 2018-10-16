======
behavior2vec
======

This package implements the ``Behavior2Vec`` algorithm introduced in the following paper.
Please cite the paper if you use this package.

Chen, Hung-Hsuan. "Behavior2Vec: Generating Distributed Representations of Users’ Behaviors on Products for Recommender Systems." *ACM Transactions on Knowledge Discovery from Data (TKDD)* 12.4 (2018).

BibTeX::

    @article{chen2018behavior2vec,
        title={Behavior2Vec: Generating Distributed Representations of Users’ Behaviors on Products for Recommender Systems},
        author={Chen, Hung-Hsuan},
        journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
        volume={12},
        issue={4},
        year={2018},
        publisher={ACM}
    }


****************************
Sample usage (with caution):
****************************

>>> import behavior2vec
>>> model = behavior2vec.Behavior2Vec()
>>> model.train('./data/sample_data.txt')  # log file, one line per user (session)
>>> model.most_similar_behavior('v-100', 'p', k=5)  # predict most similar p-type (purchasing) behavior to 'v-100' (view item 100)
>>> model.most_similar_item('100', k=5) # predict most similar items to item 100

***************
How to install:
***************

``python setup.py install``

*******************
Command line tools:
*******************

After installation, you may run the following scripts directly (tested in Ubuntu 16.04 and OS X El Capitan).

To generate the Behavior2Vec model, run:
========================================

``b2v-train.py [train-file]``

This will generate a model file of the name ``[train-file]-b2v-model.pck`` under the same directory.

To show the most similar behavior, run:
=======================================

``b2v-most-similar-behavior.py [test-file] [model-file] [output-file] [k] [behavior-type]``

The ``[test-file]`` contains a list of query behaviors, one behavior per line.

The ``[output-file]`` is the output file, which contains a list of the top ``[k]`` most simiar behaviors

To show the most similar items, run:
====================================

``b2v-most-similar-item.py [test-file] [model-file] [output-file] [k]``

The ``[test-file]`` contains a list of query items, one behavior per line.
The ``[output-file]`` is the output file, which contains a list of the top ``[k]`` most simiar items
