from common import split_classification
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
Xtr,Xte,ytr,yte=split_classification();
for m in [RandomForestClassifier(n_estimators=150,random_state=42),GradientBoostingClassifier(n_estimators=150,learning_rate=.05,random_state=42)]: m.fit(Xtr,ytr); print(type(m).__name__,accuracy_score(yte,m.predict(Xte)))