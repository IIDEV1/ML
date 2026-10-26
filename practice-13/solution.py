from common import split_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
Xtr,Xte,ytr,yte=split_classification();
for arch in ((32,),(64,32)):
 m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=arch,early_stopping=True,max_iter=500,random_state=42)).fit(Xtr,ytr); print(arch,accuracy_score(yte,m.predict(Xte)),len(m[-1].loss_curve_))