import numpy as np
import os 

from src.config import config

class preprocess_data:

    def fit(self,x,y=none):

        self.num_rows = x.shape[0]
        if len(X.shape) == 1:
            self.num_feature_dim = 1
        else:
            self.num_feature_dim = X.shape[1]

        
        if len(y.shape) == 1:
            self.target_feature_dim = 1
        else:
            self.target_feature_dim = y.shape[1]


        def transform(self,x=None,y=None):

            self.X = np.array(x).reshape(self.num_rows,self.num_feature_dim)
            self.Y = np.array(y).reshape(self.num_rows,self.target_feature_dim)
            return self.X, self.Y