#importing libraries
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from flask import Flask ,render_template ,request
from keras.models import load_model
import re

ann_model = load_model('model/ann_model.h5')

#load model
#naiveBayes = joblib.load('model/NaiveBayes_model.sav')
tfidf = joblib.load('model/tfidf_model.sav')

app= Flask(__name__)

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def getCleanReview(review):
    review = remove_html_tags(review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    return review

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        x=request.form['news']
        x=remove_html_tags(x)
        tfidf_test=tfidf.transform([x,]).toarray()
        y_pred=ann_model.predict(tfidf_test)
        if y_pred>0.5:
            y_pred='positive'
        else:
            y_pred='negative'
        result=str(y_pred).strip("['']")
        return render_template('index.html',result=result)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)