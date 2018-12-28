# RecSysModels
Here we have implemented various Recommender System algorithms for implicit feedback and sequential recommendation. These algorithms are implemented in Python and [TensorFlow](https://www.tensorflow.org). This package aims to provide clear, annotated, and efficient implementations of these algorithms along with wrapper classes and methods for easy experimentation and usage.

## Implicit Feedback
This package focuses on recommendations based on sequential and [implicit feedback](http://yifanhu.net/PUB/cf.pdf). In these settings there is no explicit numerical rating of items by users - only the record of actions they have taken. Thus there is only observed positive feedback - if a user `u` has not interacted with item `i`, it could either be because they dislike the item (negative) or they merely have not come upon this item yet (positive).

The algorithms implemented here approach the implicit feedback recommendation problem from a pairwise ranking perspective, where we assume that an item a user has interacted with should be ranked higher than an item that the user has not yet interacted with.

## Algorithms Implemented
- Bayesian Personalized Ranking (__BPR__), from ['BPR: Bayesian Personalized Ranking from Implicit Feedback'](https://arxiv.org/abs/1205.2618) (Rendle et al. 2009)
- Factorized Personalized Markov Chains (__FPMC__), from ['Factorizing personalized Markov chains for next-basket recommendation'](https://dl.acm.org/citation.cfm?id=1772773) (Rendle et al. 2010)
- __TransRec__, from ['Translation-based Recommendation'](https://arxiv.org/abs/1707.02410) (He, et al. 2017)

## Installation
`RecSysModels` is on [`PyPI`](https://pypi.org/), so you can install the package with `pip`:
```bash
$ pip install recsys_models
```

## Dependencies
- [`Python 3+`](https://www.python.org/) (3.6 may be required for Tensorflow-GPU on Windows)
- [`tensorflow`](https://www.tensorflow.org/install/) or [`tensorflow-gpu`](https://www.tensorflow.org/install/gpu)
- [`numpy`](http://www.numpy.org/)
- [`pandas`](https://pandas.pydata.org/pandas-docs/stable/index.html)
- [`Jupyter`/`JupyterLab`](https://jupyter.org/) (If you want to run the notebook)

## Sample Usage
See the [`sample_pipeline Jupyter Notebook`](https://github.com/shuyangli94/RecSysModels/blob/master/sample_pipeline.ipynb) for sample usage. In order to run this, you will need to download the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/) released in 2003 by the wonderful folks at the [GroupLens Lab](https://grouplens.org/) at the University of Minnesota.

## Interoperability
For interoperability, this package supports initializing a model with pretrained weights in the form of `numpy` arrays exported from models trained under other frameworks. Please see individual model files (e.g. [BPR](https://github.com/shuyangli94/RecSysModels/blob/master/recsys_models/models/bpr.py)) for a description of trainable variables and their shapes.


###### This package is released under [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) by [Shuyang Li](http://shuyangli.me/)