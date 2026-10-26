from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
texts=["отличный товар и быстрая доставка","ужасное качество и возврат","прекрасный сервис рекомендую","плохой товар не советую","быстро удобно отлично","ошибка и задержка"]*30; y=[1,0,1,0,1,0]*30; a,b,c,d=train_test_split(texts,y,test_size=.25,random_state=42,stratify=y); m=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression()).fit(a,c); print(classification_report(d,m.predict(b)))