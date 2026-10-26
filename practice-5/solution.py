from common import split_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
Xtr,Xte,ytr,yte=split_classification(); m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(Xtr,ytr); p=m.predict(Xte); print(confusion_matrix(yte,p)); print(classification_report(yte,p)); print("ROC-AUC",roc_auc_score(yte,m.predict_proba(Xte)[:,1]))