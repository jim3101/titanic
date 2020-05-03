from sklearn import svm
from sklearn.model_selection import cross_val_score


def fit_model(features, target):
    model = svm.SVC(kernel='poly')
    model.fit(features, target)
    return model

def make_predictions(model, test_features):
    predictions = model.predict(test_features)
    return predictions

def evaluate_model(features, target):
    model = svm.SVC(kernel='poly')
    scores = cross_val_score(model, features, target, cv=10)
    print('Estimated mean score: {:.2f}%'.format(scores.mean() * 100))