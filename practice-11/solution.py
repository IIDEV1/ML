from common import classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
X,y=classification(); Z=StandardScaler().fit_transform(X); p=PCA().fit(Z); cum=np.cumsum(p.explained_variance_ratio_); k=int(np.argmax(cum>=.9)+1); print("components for 90%",k); print("2D variance",cum[1]);