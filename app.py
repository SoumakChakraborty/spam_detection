import numpy as np
import pickle
import os
from flask import Flask,render_template,redirect,request,url_for

app=Flask(__name__)

@app.route('/',methods=["GET","POST"])
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

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
