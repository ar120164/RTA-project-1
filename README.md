# RTA-project-1
## Flask application with Perceptron model and Iris data
This script contains one web service: 
* **getPredict()**: returns a prediction based on petal and sepal lenghts. 
### Required libraries
First we need to install the required libraries.
* Flask
* Flask_restful
* numpy
* pandas
* scikit-learn
### Web server
First, we create the Perceptron class. Then we create a web service under the following URL: localhost:5000/?sepal={*insert your sepal lenghts here, each number separated by coma. Decimals are with .*}&petal={*insert your petal lenghts here, each number separated by coma. Decimals are with*.}. Notice that the each sepal value must corrispond to the respective petal value. So, you should insert them in the same order. 
The following URL is an example to access the web service: http://127.0.0.1:5000/?sepal=3.5,4.7,6.7&petal=2.1,1.3,9.8
The web service results are formated in HTML code. We create a table that shows the values of the features and the corresponding predicted class. 
