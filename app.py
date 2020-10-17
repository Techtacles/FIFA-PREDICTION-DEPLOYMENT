from flask import Flask,request, jsonify, render_template
import pickle
import numpy as np
#import os
#from pathlib import Path
app = Flask(__name__)
#dire='C:\\Users\\USER\\Desktop\\NEW DEPLOYMENT\\model.pkl'

model=pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    pred=prediction[0]
    pred1=f"{pred:,}"
    pred2=pred1.split(".")[0]
   #output=round(prediction[0],2)
    return render_template("index.html",prediction_text="The value of the player is $ {}".format(pred2))
if __name__ == '__main__':
    app.run(debug=True)