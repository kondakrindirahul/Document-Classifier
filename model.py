import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def trainModel():
    # load the csv data as a pandas data frame
    data = pd.read_csv('shuffled-full-set-hashed.csv', header=None, names=['label', 'hashed-document'])

    # remove rows with missing values
    data.dropna(inplace=True)

    # split the given data into training and test data
    train_docs, test_docs, train_labels, test_labels = train_test_split(data['hashed-document'].values,
                                                                        data['label'].values, test_size=0.2)

    # transform the documents into tf-idf vectors
    vectorizer = TfidfVectorizer()
    fitted_vectorizer = vectorizer.fit_transform(train_docs)

    # use pickle to save the vectorizer to disk
    pickle.dump(vectorizer, open('vectorizer.pkl', "wb"))

    # fit the random forest classification model on the training data
    rfClassifier = RandomForestClassifier(n_estimators=50, min_samples_split=10, bootstrap=True, class_weight='balanced')
    rfClassifier.fit(fitted_vectorizer, train_labels)
    predicted_labels = rfClassifier.predict(vectorizer.transform(test_docs))

    # use pickle to save the classifier to disk
    joblib.dump(rfClassifier, 'rfClassifier.joblib')

    print("Training Data Accuracy: ", metrics.accuracy_score(train_labels, rfClassifier.predict(fitted_vectorizer)))
    print("Test Data Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))

    confusion_matrix = pd.DataFrame(metrics.confusion_matrix(test_labels, predicted_labels), columns=rfClassifier.classes_, index=rfClassifier.classes_)
    confusion_matrix.to_csv('confusion_matrix.csv')


if __name__ == '__main__':
    trainModel()
