from common import split_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
Xtr,Xte,ytr,yte=split_classification();
for C in (.1,1,10):
 m=make_pipeline(StandardScaler(),SVC(C=C,gamma='scale')).fit(Xtr,ytr); print(C,accuracy_score(yte,m.predict(Xte)))