import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def np_mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def np_root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np_mean_squared_error(y_true, y_pred))

def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np_absolute_error(y_true, y_pred))

def np_standard_deviation(y_true, y_pred):
    return np.std(y_true - y_pred)

def np_absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred)

def calculate_error(y_true, y_pred):
    mse = np_mean_squared_error(y_true, y_pred)
    rmse = np_root_mean_squared_error(y_true, y_pred)
    mae = np_mean_absolute_error(y_true, y_pred)
    std = np_standard_deviation(y_true, y_pred)
    ae = np_absolute_error(y_true, y_pred)

    return mse, rmse, mae, std, ae

# Alternative way to calculate metrics using sklearn
def calculate_error_sklearn(y_true, y_pred):
    mse_val = mean_squared_error(y_true, y_pred)
    rmse_val = root_mean_squared_error(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    mape_val = mean_absolute_percentage_error(y_true, y_pred)
    
    return mse_val, rmse_val, mae_val, mape_val