from common import regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
X,y=regression(); a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42); m=LinearRegression().fit(a,c); p=m.predict(b); print({"MAE":mean_absolute_error(d,p),"RMSE":mean_squared_error(d,p)**.5,"R2":r2_score(d,p)}); print("largest coefficients",X.columns[np.argsort(abs(m.coef_))[-3:]].tolist())