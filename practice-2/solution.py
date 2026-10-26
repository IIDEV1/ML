import pandas as pd, matplotlib.pyplot as plt
from common import classification
X,y=classification(); print(X.info()); print(X.describe().T); print("missing",X.isna().sum().sum()); print("target",y.value_counts().to_dict()); print(X.corr().abs().stack().sort_values(ascending=False).head(12)); X.iloc[:,:4].hist(figsize=(8,6)); plt.tight_layout(); plt.savefig("eda.png")