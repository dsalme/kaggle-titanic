## Simple Flask + Vue app that serves the [Titanic ML competition Model](https://risx3.github.io/titanic-analysis/)
### Submit your variables for running predictions using the Titanic ML Model

It was trained by following [this guide](https://risx3.github.io/titanic-analysis/)

### Endpoints
- `/`
It's the root url, which returns a frontend view
- `/submit_prediction`
Lets you submit a csv file for running model predictions.
The csv file should have the following columns:
`Age,SibSp,Parch,Fare,male,Q,S,2,3`
And should have exactly 268 (plus 1 for the headers, so 269) rows.

### How to run it
The app is composed by one docker container, to run it, run this in your console after cloning it:
- `docker-compose up --build`

### How its structured
The app consists in a Flask App, which serves a web app through the `/` route.
The frontend app is a simple index.html file which loads VueJS through ES modules, and has a VueJS form component for handling the csv submission and showing the `/submit_prediction`'s endpoint response.