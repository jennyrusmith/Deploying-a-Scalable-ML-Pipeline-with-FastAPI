import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import apply_label
from ml.model import train_model, compute_model_metrics

def test_apply_label():
    """
    This test checks if the binary predictions
    are converted to the correct labels.
    """
    assert apply_label([1]) == ">50K"  # checks that 1 is converted to >50K
    assert apply_label([0]) == "<=50K"  # checks that 0 is converted to <=50K

def test_train_model():
    """
    This test checks if train_model returns
    a trained Random Forest classifier.
    """
    X = np.array([[1,2], [2,3], [3,4], [4,5]]) # sample feature data
    y = np.array([0,0,1,1]) # sample target labels

    model = train_model(X, y) # train the model using the sample data

    assert isinstance(model, RandomForestClassifier) # checks that the model is a Random Forest


def test_compute_model_metrics():
    """
    This test checks that the model metrics
    return the expected values.
    """
    y = np.array([0,0,1,1]) # actual labels
    preds = np.array([0,1,1,1]) # predicted labels

    precision, recall, fbeta = compute_model_metrics(y, preds) # calculate the metrics

    assert precision == pytest.approx(2/3) # checks that precision is correct
    assert recall == pytest.approx(1.0) # checks that recall is correct
    assert fbeta == pytest.approx(0.8)  # checks that F1 is correct

