from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='iqr', factor=1.5):
        self.method = method
        self.factor = factor
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            raise ValueError("Input must be a pandas DataFrame")
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.method == 'iqr':
            mask = pd.Series(True, index=X.index)
            for col in self.columns_:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.factor * IQR
                upper = Q3 + self.factor * IQR
                mask &= X[col].between(lower, upper)
            return X[mask]
        else:
            raise ValueError("Only 'iqr' method is supported for now.")
