Sources Used 
{
  https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard, 
  https://www.kaggle.com/sungsujaing/blackfriday-eda-randomforestprediction
}
import numpy as np ( It provides fast and efficient operations on arrays of homogeneous data)

import pandas as pd ( It is a software library written for the Python programming language for data manipulation and analysis)

%matplotlib inline

import matplotlib.pyplot as plt ( It is a plotting library for the Python programming language and its numerical mathematics) extension NumPy.

import seaborn as sns ( It is a Python data visualization library based on matplotlib. It provides a high-level interface for   drawing attractive and informative statistical graphics)

color = sns.color_palette()

sns.set_style('darkgrid')

from math import sqrt ( Used to calculate Square Root)

from sklearn import preprocessing (It is a package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators)

from sklearn.cross_validation import KFold, cross_val_score (KFold Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default), cross_val_score evaluate a score by cross-validation)

from sklearn.metrics import mean_squared_error ( Mean squared error regression loss)

from scipy import stats (This module contains a large number of probability distributions as well as a growing library of statistical functions)

from scipy.stats import boxcox (Return a positive dataset transformed by a Box-Cox power transformation)

from scipy.stats import norm, skew (A normal continuous random variable, Compute the skewness of a data set.)

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV (Exhaustive search over specified parameter values for an estimator.)

from sklearn.model_selection import learning_curve (plots learning curve)
