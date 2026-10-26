from common import split_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,StratifiedKFold
Xtr,Xte,ytr,yte=split_classification(); pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)); cv=StratifiedKFold(5,shuffle=True,random_state=42); grid=GridSearchCV(pipe,{"logisticregression__C":[.01,.1,1,10,100]},cv=cv,scoring="accuracy").fit(Xtr,ytr); print(grid.best_params_,grid.best_score_,grid.score(Xte,yte))