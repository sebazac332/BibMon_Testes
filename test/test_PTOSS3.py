import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from bibmon import sklearnRegressor

@pytest.fixture
def dummy_data():
    X = pd.DataFrame(np.random.rand(10, 3), columns=["feat1", "feat2", "feat3"])
    Y = pd.DataFrame(np.random.rand(10, 1), columns=["target"])
    return X, Y

def create_mock_model(**attributes):
    return SimpleNamespace(**attributes)


# CT1
def test_plot_importances_ct1(dummy_data):
    X, Y = dummy_data
    model = create_mock_model(coef_=np.array([0.1, 0.2, 0.3]), perm_feature_importances_=np.array([0.3, 0.4, 0.3]))
    reg = sklearnRegressor(model, permutation_importance=True)
    reg.X_train = X
    reg.Y_train = Y
    df = reg.plot_importances(permutation_importance=True)
    assert isinstance(df, pd.Series)
    assert not df.empty


# CT2
def test_plot_importances_ct2(dummy_data):
    X, Y = dummy_data
    model = create_mock_model(coef_=np.array([0.1, 0.2, 0.3]))
    reg = sklearnRegressor(model, permutation_importance=False)
    reg.X_train = X
    reg.Y_train = Y
    df = reg.plot_importances(permutation_importance=False)
    assert isinstance(df, pd.Series)
    assert not df.empty


# CT3
def test_plot_importances_ct3(dummy_data):
    X, Y = dummy_data
    model = create_mock_model(feature_importances_=np.array([0.5, 0.3, 0.2]), 
                              perm_feature_importances_=np.array([0.3, 0.3, 0.4]))
    reg = sklearnRegressor(model, permutation_importance=True)
    reg.X_train = X
    reg.Y_train = Y
    df = reg.plot_importances(permutation_importance=True)
    assert isinstance(df, pd.Series)
    assert not df.empty


# CT4
def test_plot_importances_ct4_no_importances(dummy_data, capsys):
    X, Y = dummy_data
    model = create_mock_model()
    reg = sklearnRegressor(model, permutation_importance=True)
    reg.X_train = X
    reg.Y_train = Y
    result = reg.plot_importances(permutation_importance=True)
    captured = capsys.readouterr()
    assert "There are no importances" in captured.out
    assert result is None


# CT5
def test_plot_importances_ct5_no_importances_false(dummy_data, capsys):
    X, Y = dummy_data
    model = create_mock_model()
    reg = sklearnRegressor(model, permutation_importance=False)
    reg.X_train = X
    reg.Y_train = Y
    result = reg.plot_importances(permutation_importance=False)
    captured = capsys.readouterr()
    assert "There are no importances" in captured.out
    assert result is None


# CT6
def test_plot_importances_ct6_coef_only(dummy_data):
    X, Y = dummy_data
    model = create_mock_model(coef_=np.array([0.1, 0.2, 0.3]))
    reg = sklearnRegressor(model)
    reg.X_train = X
    reg.Y_train = Y
    df = reg.plot_importances()
    assert isinstance(df, pd.Series)
    assert not df.empty


# CT7
def test_plot_importances_ct7_feature_importance_only(dummy_data):
    X, Y = dummy_data
    model = create_mock_model(feature_importances_=np.array([0.4, 0.3, 0.3]))
    reg = sklearnRegressor(model)
    reg.X_train = X
    reg.Y_train = Y
    df = reg.plot_importances()
    assert isinstance(df, pd.Series)
    assert not df.empty


# CT8
def test_plot_importances_ct8_perm_only(dummy_data):
    X, Y = dummy_data
    model = create_mock_model(perm_feature_importances_=np.array([0.2, 0.4, 0.4]))
    reg = sklearnRegressor(model)
    reg.X_train = X
    reg.Y_train = Y
    df = reg.plot_importances()
    assert isinstance(df, pd.Series)
    assert not df.empty
