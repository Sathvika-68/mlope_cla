
from model import train_model
from sklearn.datasets import load_iris


def test_model_trains_successfully():
    model, accuracy = train_model()
    assert model is not None
    assert 0.0 <= accuracy <= 1.0


def test_model_prediction():
    model, _ = train_model()
    iris = load_iris()

    prediction = model.predict([iris.data[0]])

    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]


def test_iris_dataset():
    iris = load_iris()

    assert iris.data.shape == (150, 4)
    assert len(iris.target) == 150
