from common import split_classification
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
Xtr,Xte,ytr,yte=split_classification(); num=list(Xtr.columns); pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num)])
A=pre.fit_transform(Xtr); B=pre.transform(Xte); print(A.shape,B.shape); print("fit only on train: OK")