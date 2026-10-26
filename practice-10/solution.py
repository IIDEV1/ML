from common import classification
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X,y=classification(); Z=StandardScaler().fit_transform(X)
for k in (2,3,4,5):
 labels=KMeans(k,n_init=10,random_state=42).fit_predict(Z); print(k,silhouette_score(Z,labels))