from common import split_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
Xtr,Xte,ytr,yte=split_classification();
for m in [DecisionTreeClassifier(max_depth=4,random_state=42),RandomForestClassifier(n_estimators=200,max_depth=6,random_state=42,n_jobs=-1)]: m.fit(Xtr,ytr); print(type(m).__name__,accuracy_score(yte,m.predict(Xte)))
