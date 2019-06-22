{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# supervised machine learning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np #for array \n",
    "import pandas as pd # for data loading\n",
    "from sklearn.tree  import DecisionTreeClassifier # only decision tree classifier for classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# here we do the classificatin between apple and orange\n",
    "# here apple is smooth and orange is bumpy\n",
    "# 0 for smooth and  1 for bumpy\n",
    "features = [[100,0],[120,0],[130,1],[150,1]]  # these are features of  aplle and orange first is wait and second is smooth or bumpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "labels = [\"apple\",\"apple\",\"orange\",\"orange\"] # so this the answer for each feature that we teach the machine throgh DecisionTreeClassifier algo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "dtc=DecisionTreeClassifier() # calling  the DecisionTreeClassifier algo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "trained = dtc.fit(features,labels) # fit the data in algo and traine the machine for prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['orange', 'orange'], dtype='<U6')"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trained.predict([[127,0],[126,1])  # prediction of labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
