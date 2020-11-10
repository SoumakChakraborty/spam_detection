import numpy as np
import pickle
from flask import Flask,render_template,redirect,request,url_for

app=Flask(__name__)

@app.route('/detect',methods=["GET","POST"])
def detect():
    if request.method=="GET":
        return render_template('index.html')
    else:
        msg=request.form.get("message")
        model=pickle.load(open('spam.pkl','rb'))
        msg=msg.lower()
        msg=msg.replace('^a-zA-Z','')
        X_r=[msg]
        TF=pickle.load(open('vectorizer.pkl','rb'))
        X=TF.transform(X_r).toarray()
        Y=model.predict(X)
        if Y[0]==0:
            return render_template('index.html',message='Not Spam')
        else:
            return render_template('index.html',message='Spam')

app.run(debug=True)