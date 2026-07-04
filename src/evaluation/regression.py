'''
Build metrics for Regression
'''

import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Calculate how large the error is between y_true and y_pred
    
    Objective:
        Plus -> Make error sensitive and make model learn fast
        Minus -> If there is outlier can explode
    
    Formula:
        mse = mean(pow(y_true - y_pred))
    '''
    error = y_true - y_pred
    squared_error = error**2
    mse = np.mean(squared_error)
    return float(mse)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Same as mse but with root
    
    Objective:
        Cover minus from MSE, np.sqrt back again scale feature into normal scale while keep
        sensivity error towards outlier
    
    Formula:
        rmse = np.sqrt(mse)
    '''
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse) # equal to mse ** 0.5
    return rmse

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Calculate error in absolute version
    
    Objective:
        Plus -> Robust towards outlier, and keep scale stable
        Minus -> Too long for training learn, because move linearly 
    
    Formula:
        mae = mean(|y_true - y_pred|)
    '''
    error = y_true - y_pred
    abs_error = np.abs(error)
    mae = np.mean(abs_error)
    return float(mae)