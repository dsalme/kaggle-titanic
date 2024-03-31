import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import logging

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


train = pd.read_csv('/src/kaggle/input/titanic/train.csv')
test = pd.read_csv('/src/kaggle/input/titanic/test.csv')

# train
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
pcl = pd.get_dummies(train['Pclass'],drop_first=True)

train = pd.concat([train,sex,embark,pcl],axis=1)

train.drop(['Pclass','Sex','Embarked','Cabin','PassengerId','Name','Ticket'],axis=1, inplace=True)

train_values = {'Age': round(np.mean(train['Age']))}
train = train.fillna(value = train_values)

# test
sex = pd.get_dummies(test['Sex'], drop_first = True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
pcl = pd.get_dummies(test['Pclass'],drop_first=True)

test = pd.concat([test,sex,embark,pcl],axis=1)

test.drop(['Pclass','Sex','Embarked','Cabin','PassengerId','Name','Ticket'],axis=1, inplace=True)

test_values = {'Age':round(np.mean(test['Age'])), 'Fare':round(np.mean(test['Fare']))}
test = test.fillna(value = test_values)

X = train.drop('Survived',axis=1)
X.columns = X.columns.astype(str)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
output_vars = X_test
output_vars.to_csv('test_vars.csv', index=False)

logmodel = LogisticRegression(solver = 'liblinear')

logmodel.fit(X_train, y_train)
joblib.dump(logmodel, 'modelo_titanic.pkl')

predictions = logmodel.predict(X_test)

my_score = accuracy_score(y_test, predictions)

app.logger.info(f"my score: {my_score}")

logmodel = joblib.load("modelo_titanic.pkl")

@app.route('/submit_variables', methods=['POST'])
def submit_predict():
    if 'uploaded_file' not in request.files:
        return jsonify({
            "error":f'No upload_file provided for the submit_variables endpoint'
        }), 400

    uploaded_file = request.files['uploaded_file']
    if uploaded_file.filename == '':
        return jsonify({"error": "uploaded_file is invalid"}), 400

    if uploaded_file:
        test_vars_df = pd.read_csv(uploaded_file)
        uploaded_predict = logmodel.predict(test_vars_df)
        accuracy = accuracy_score(y_test, uploaded_predict)
        app.logger.info(f"uploaded file accuracy test: {accuracy}")
        return jsonify({
            "data":{
                "predictions": uploaded_predict.tolist(),
                "accuracy": f'{accuracy}'
            }
        })

@app.route('/')
def index():
    return render_template('index.html', message="lalala")
