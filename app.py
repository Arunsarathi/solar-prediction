import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.ensemble import ExtraTreesRegressor
import math
import joblib

app = Flask(__name__)
#model = pickle.load( open('solar.pkl','rb'))
model = joblib.load('solar.pkl')

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/predict', methods =['POST'])
def predict():

    
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    # we get two sqr "[[]]" - one is int_features,
    # another is final_features
    result = model.predict(final_features)
    
        #print('you dont have diabetes')
    return render_template('index.html',prediction_text= result )

    




if __name__ == '__main__':
    app.run(debug = True)
    #app.run(host='0.0.0.0', port=8080)

# http://localhost:5000/


