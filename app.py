from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
#import sklearn.externals.joblib as extjoblib
import joblib
import pickle

# load the model from disk
spam_detect_model = open('pickle1.pkl','rb')
clf = joblib.load(spam_detect_model)
cv_model = open('transform.pkl', 'rb')
cv = joblib.load(cv_model)
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)