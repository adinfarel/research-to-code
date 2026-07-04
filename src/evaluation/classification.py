'''
Build metric for Classification
'''

import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    # true_pred = y_true == y_pred
    # false_pred = y_true != y_pred
    
    # TP_mask = y_pred[true_pred] == 1
    # FP_mask = y_pred[false_pred] == 1
    # TN_mask = y_pred[true_pred] == 0
    # FN_mask = y_pred[false_pred] == 0
    
    # TP = TP_mask.sum()
    # FP = FP_mask.sum()
    # TN = TN_mask.sum()
    # FN = FN_mask.sum()
    
    TP = np.sum((y_true == 1) & (y_pred == 1)) 
    FP = np.sum((y_true == 0) & (y_pred == 1)) 
    TN = np.sum((y_true == 0) & (y_pred == 0)) 
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    cm = np.array([[TP, FP], [TN, FN]])
    
    return cm

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    INTUITION:
    How good model doing prediction, but accuracy score can deceive
    Accuracy score going to worse if we dealing with imbalance data
    Imagine if we get 99% accuracy score in 490 healthy patient data and 10 diseased patient data
    the model will only make a lazy prediction bcs if it finds that most of the data is healthy
    the model will keep guessing healthy
    '''
    cm = confusion_matrix(y_true, y_pred)
    
    acc = (cm[0][0] + cm[1][0]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1] + 1e-8)
    
    return acc

def precision(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    INTUITION:
    out of all the times the model predicts "true", how many r actually "true"?
    this prevent model trying to predict false positive
    '''
    cm = confusion_matrix(y_true, y_pred)
    
    prec = cm[0][0] / (cm[0][0] + cm[0][1] + 1e-8)
    
    return prec

def recall(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    INTUITION:
    out of all the labels "true", how many model can catch "true"?
    this prevent model trying to predict false negative
    '''
    cm = confusion_matrix(y_true, y_pred)
    
    rec = cm[0][0] / (cm[0][0] + cm[1][1] + 1e-8)
    
    return rec
    
def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    f1_ = (2 * prec * rec) / (prec + rec + 1e-8)
    return f1_