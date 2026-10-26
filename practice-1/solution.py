from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
d=load_breast_cancer(as_frame=True); Path("data").mkdir(exist_ok=True)
d.frame.to_csv("data/cancer.csv",index=False); print(d.frame.shape); print(d.frame.dtypes.value_counts()); print(d.frame.isna().sum().sum()); print(d.frame.describe().T.head())