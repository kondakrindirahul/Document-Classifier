from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
from sklearn.externals import joblib

app = Flask(__name__)

# predict the class for input hashed document
def getPredictionConfidence(document):
    # load the saved tf-idf vectorizer from pickle file
    with open('vectorizer.pkl', 'rb') as myFile:
        vectorizer = pickle.load(myFile)
    # convert input hashed document into a vector
    fitted_vectorizer = vectorizer.transform([document])
    # load the trained model from the pickle file
    rfClassifier = joblib.load('rfClassifier.joblib')
    # get the predicted label for input document
    labelPredicted = rfClassifier.predict(fitted_vectorizer)

    # calculate confidence: max of predicted probabilities
    predictedProbabilities = rfClassifier.predict_proba(fitted_vectorizer)[0]
    confidence = max(predictedProbabilities)


    return labelPredicted[0].title(), np.round(confidence, decimals=2) * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        hashed_doc = request.form['doc']
        label, confidence = getPredictionConfidence(hashed_doc)
        return render_template("HomePage.html", label=label, confidence=confidence)

    if request.method == 'GET':
        if 'words' in request.args:
            input_doc = request.args.get('words')

            label, confidence = getPredictionConfidence(input_doc)
            data = {'prediction': label,
                    'confidence': str(confidence) + '%'
                    }
            response = jsonify(data)
            response.status_code = 200

            return response
        else:
            return render_template('HomePage.html')

    return render_template("HomePage.html")


if __name__ == '__main__':
    app.run()