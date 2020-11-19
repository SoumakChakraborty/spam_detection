import numpy as np
import pandas as pd
from sklearn import svm,ensemble
from sklearn import metrics,preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import cross_val_score
import pickle
from sklearn import feature_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('spam.csv',encoding='iso-8859-1',usecols=['v1','v2'])



df['v2']=df['v2'].str.lower()



df['v2']=df['v2'].str.replace('^a-zA-Z',"")



df['v1'].replace('ham',0,inplace=True)


df['v1'].replace('spam',1,inplace=True)




Y=df['v1']




X=df['v2']




RM=RandomOverSampler(random_state=42)




TF=TfidfVectorizer(ngram_range=(1,2),strip_accents=None,stop_words='english')



X_f=TF.fit_transform(X)

X_samp,Y_samp=RM.fit_sample(X_f,Y)




#X_train,X_test,Y_train,Y_test=train_test_split(X_samp,Y_samp,train_size=.70)


model=MultinomialNB()

model.fit(X_samp,Y_samp)

pickle.dump(model,open('spam.pkl','wb'))
pickle.dump(TF,open('vectorizer.pkl','wb'))



