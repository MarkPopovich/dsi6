#!/Users/mark/anaconda/envs/dsi/bin/python
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class ClassificationMetrics:  
    def __init__(self):
        pass
    
    def fit(self, y_true, y_pred):
        self.cm = confusion_matrix(y_true, y_pred)
        self.tn, self.fp, self.fn, self. tp = self.cm.ravel()
        self.confusion_matrix = pd.DataFrame(self.cm, 
                                             columns = ['predicted negative', 'predicted positive'], 
                                             index = ['actual negative', 'actual positive'])
        
        self.accuracy_ = (self.tn + self.tp) / len(y_true)
        self.misclassification_ = 1 - self.accuracy_
        self.sensitivity_ = self.tp / (self.tp + self.fn)
        self.specificity_ = self.tn / (self.tn + self.fp)
        self.precision_ = self.tp / (self.tp + self.fp)
        self.negative_predictive_value_ = self.tn / (self.tn + self.fn)
        self.false_positive_rate_ = 1 - self.specificity_
        self.false_negative_rate_ = self.fp / (self.fp + self.tn)
        
        
    def describe(self):
        return pd.DataFrame({'Metric' : {'Accuracy' : self.accuracy_,
                           'Misclassification' : self.misclassification_,
                           'Sensitivity' : self.sensitivity_,
                           'Specificity' : self.specificity_,
                           'Precision' : self.precision_,
                           'Negative Predictive Value' : self.negative_predictive_value_,
                           'False Positive Rate' : self.false_positive_rate_,
                           'False Negative Rate' : self.false_negative_rate_
                            }})
    
    
class RegressionMetrics:
    mse_ = None
    rmse_ = None
    r2_ = None
    base_mse_ = None
    base_rmse_ = None
    
    r2_adj_ = None
    msle_ = None
    mae_ = None
    
    def __init__(self):
        pass
        
    def fit(self, y_true, y_pred, k = None):
        self.mse_ = self.calculate_mse(y_true, y_pred)
        self.rmse_ = self.calculate_rmse(y_true, y_pred)
        self.r2_ = self.calculate_r2(y_true, y_pred)
        self.msle_ = self.calculate_msle(y_true, y_pred)
        self.mae_ = self.calculate_mae(y_true, y_pred)
        self.base_mse_ = self.baseline_mse(y_true)
        self.base_rmse_ = self.baseline_rmse(y_true)
        
        if k:
            self.r2_adj_ = self.calculate_r2_adj(y_true, y_pred, k)
        
    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def calculate_rmse(self, y_true, y_pred):
        return self.calculate_mse(y_true, y_pred) ** 0.5
    
    def calculate_r2(self, y_true, y_pred):
        return 1 - np.sum(self.calculate_mse(y_true, y_pred)) / \
                   np.sum(self.baseline_mse(y_true))
    
    def calculate_r2_adj(self, y_true, y_pred, k):
        n = len(y_true)
        return 1 - (1 - self.r2_) * (n - 1) / (n - k - 1)
    
    def calculate_msle(self, y_true, y_pred):
        return np.mean(np.log(np.abs(y_true - y_pred)) ** 2)
        
    def calculate_mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    
    def baseline_mse(self, y_true):
        return self.calculate_mse(y_true, np.mean(y_true))
        
    def baseline_rmse(self, y_true):
        return self.calculate_rmse(y_true, np.mean(y_true))

    def describe(self):
        output = {var : vars(self)[var] for var in vars(self)}
        description = {'Metrics' : {
                           'MSE' : self.mse_,
                           'RMSE' : self.rmse_,
                           'Baseline MSE' : self.base_mse_,
                           'Baseline RMSE' : self.base_rmse_,
                           'R2' : self.r2_,
                       'R2 Adjusted' : self.r2_adj_,
                           'MAE' : self.mae_,
                           'MSLE' : self.msle_
                           }
                      }
        return pd.DataFrame(description)