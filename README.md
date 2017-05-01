## vantgrd (for Python)

# Introduction

In the last year, I have rediscover an interesting in machine learning,
many years after the "Artificial Intelligence" course at the University
of Pisa. So I took the chance the implement a few basic ML algorithms,
like Logistic Regression and Factorization Machine using Python, using
this project as a playground to learn Python as well.

I believe this repo will be interesting for those people that like to
understand not only what works, but also why it works. **Using a more
established ML Library for a production workload is definitely a better
idea** (weka, mllib, sklearn and so on), but understand what's going on
inside it is even better, in my opinion.

# Available algorithms

- Logistic Regression
    * with Adagrad optimiser
    * with Adadelta optimiser
    * with FTRL Proximal optimiser
- Factorization Machine (Binary Classification)
    * with Stochastic Gradient Descent optimiser
    * with Adadelta optimiser

# Datasets

- http://archive.ics.uci.edu/ml/datasets.html

# References and interesting links

- http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
- https://github.com/kastnerkyle/kaggle-criteo/blob/master/clf.py
- https://gist.github.com/ceshine/f7f93046c58fe6ee840b
- http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
- https://github.com/fmfn/FTRLp/blob/master/FTRLp.py
- https://github.com/saiias/Adadelta/
- http://www.libfm.org
- https://github.com/srendle/libfm