import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

def classification():
 d=load_breast_cancer(as_frame=True); return d.data,d.target

def regression():
 d=load_diabetes(as_frame=True); return d.data,d.target

def split_classification():
 X,y=classification(); return train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
