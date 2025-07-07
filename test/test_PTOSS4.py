import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from bibmon import sklearnRegressor

@pytest.fixture
def dummy_data():
    X = pd.DataFrame(np.random.rand(10, 3), columns=["f1", "f2", "f3"])
    Y = pd.DataFrame(np.random.rand(10, 1), columns=["target"])
    return X, Y

def create_mock_model(**attrs):
    return SimpleNamespace(**attrs)

def test_plot_importances_invalid_permutation_type(dummy_data, capsys):
    X, Y = dummy_data

    model = create_mock_model(coef_=np.array([0.2, 0.4, 0.4]))

    reg = sklearnRegressor(model)
    reg.X_train = X
    reg.Y_train = Y

    result = reg.plot_importances(permutation_importance="yes")

    assert result is None or isinstance(result, type(None))


def test_plot_importances_invalid_permutation_type_string(dummy_data, capsys):
    X, Y = dummy_data

    model = create_mock_model(coef_=np.array([0.2, 0.4, 0.4]))

    reg = sklearnRegressor(model)
    reg.X_train = X
    reg.Y_train = Y

    result = reg.plot_importances(permutation_importance="macarrao")

    captured = capsys.readouterr()

    assert result is None or isinstance(result, type(None))
    assert "Erro: valor de permutation_importance é 'macarrao'" in captured.out
    assert "O tipo de dado String nao é valido para essa variavel" in captured.out
    assert "Tipo de dado esperado: bool (True ou False)" in captured.out

def test_plot_importances_invalid_permutation_type_int(dummy_data, capsys):
    X, Y = dummy_data

    model = create_mock_model(coef_=np.array([0.2, 0.4, 0.4]))

    reg = sklearnRegressor(model)
    reg.X_train = X
    reg.Y_train = Y

    result = reg.plot_importances(permutation_importance=23)

    captured = capsys.readouterr()

    assert result is None or isinstance(result, type(None))
    assert "Erro: valor de permutation_importance é '23'" in captured.out
    assert "O tipo de dado Integer nao é valido para essa variavel" in captured.out
    assert "Tipo de dado esperado: bool (True ou False)" in captured.out

def test_plot_importances_invalid_permutation_type_float(dummy_data, capsys):
    X, Y = dummy_data

    model = create_mock_model(coef_=np.array([0.2, 0.4, 0.4]))

    reg = sklearnRegressor(model)
    reg.X_train = X
    reg.Y_train = Y

    result = reg.plot_importances(permutation_importance=2.55)

    captured = capsys.readouterr()

    assert result is None or isinstance(result, type(None))
    assert "Erro: valor de permutation_importance é '2.55'" in captured.out
    assert "O tipo de dado Float nao é valido para essa variavel" in captured.out
    assert "Tipo de dado esperado: bool (True ou False)" in captured.out