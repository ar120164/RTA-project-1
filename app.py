import pandas as pd
import numpy as np
from flask_restful import Resource, Api
from flask import Flask
from flask import request, jsonify
from sklearn.datasets import load_iris

#First we define the perceptron class containing the model
class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)
    
#We import and format the iris data
iris = load_iris()
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                 columns = iris['feature_names']+ ['target']) 
X, y = df.iloc[:100, [0,2]].values, df.iloc[:100, 4]
def a(x):
    if x == 0:
        return -1
    else:
        return 1
y = y.map(a)

#We create a instance of Perceptron class
model = Perceptron()

#We train the model with the data
model.fit(X, y)

#We create the web services

app = Flask(__name__)
@app.route('/', methods=['GET'])
def getPredict():
    global model
    sepal_lengths = request.args.get("sepal", [])
    petal_lengths = request.args.get("petal", [])
    if(sepal_lengths == None or len(sepal_lengths) == 0):
        return "Error! sepal should not be empty. For testing the web server try: http://127.0.0.1:5000/?sepal=3.5,4.7,6.7&petal=2.1,1.3,9.8"
    
    if(petal_lengths == None or len(petal_lengths) == 0):
        return "Error! petal should not be empty. For testing the web server try: http://127.0.0.1:5000/?sepal=3.5,4.7,6.7&petal=2.1,1.3,9.8"
    
    html_text = "<table border=\"1\"><caption>Ana Gabriela - Iris data and Perceptron model</caption>"
    html_text += '''<thead>
                        <tr>
                            <th colspan=\"2\">Features</th>
                            <th rowspan=\"2\">Predicted class</th>
                        </tr>
                        <tr>
                            <th>Sepal lenght</th>
                            <th>Petal lenght</th>
                        </tr>
                    </thead>'''
    
    sepal = sepal_lengths.split(',')
    sepal = np.array(sepal)
    sepal = sepal.astype(float)
    
    petal = petal_lengths.split(',')
    petal = np.array(petal)
    petal = petal.astype(float)
    
    if(len(petal) != len(sepal)):
        return "Error! petal and sepal should have the same dimension. For testing the web server try: http://127.0.0.1:5000/?sepal=3.5,4.7,6.7&petal=2.1,1.3,9.8"
    
    predict_arg  = [[sepal[0], petal[0]]]
    
    for i in range (1, len(sepal)):
        predict_arg  = np.append(predict_arg, [[sepal[i], petal[i]]], axis=0)
        
    predictions = model.predict(predict_arg)
    
    html_text +="<tbody>"
    for i in range(0, len(predict_arg)):
        html_text += "<tr><td>"+str(predict_arg[i][0])+"</td><td>"+str(predict_arg[i][1])+"</td><td>"+str(predictions[i])+"</td></tr>"
    html_text +="</tbody></table>"

    return str(html_text)

app.run()