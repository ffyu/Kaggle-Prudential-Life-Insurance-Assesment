# Prudential - my solution
import time
import numpy as np
import pandas as pd
from scipy.optimize import fmin_powell
from sklearn.cross_validation import StratifiedKFold
from ml_metrics import quadratic_weighted_kappa
import xgboost as xgb


start_time = time.time()
FOLDER = './data/'
FILE1 = 'train_processed.csv'
FILE2 = 'test_processed.csv'


def digitize(y_pred, splits):
    
    if y_pred < splits[0]:
        return 1
    elif y_pred < splits[1]:
        return 2
    elif y_pred < splits[2]:
        return 3
    elif y_pred < splits[3]:
        return 4
    elif y_pred < splits[4]:
        return 5
    elif y_pred < splits[5]:
        return 6
    elif y_pred < splits[6]:
        return 7
    else:
        return 8


def qwk_wrapper(y, y_pred, splits):
            
    return quadratic_weighted_kappa([digitize(yp, splits) for yp in y_pred], y)
    

def splits_opt(y, y_pred, splits_guesses):
    
    result = fmin_powell(lambda s: -qwk_wrapper(y, y_pred, s), 
                         splits_guesses, disp=False)
    
    return result


def fill_null(train, test):
    
    for df in [train, test]:
        df.fillna(value=-1000, inplace=True)
    
    return train, test
    
    
class PrudentialModel():
    
    def __init__(self, params_lst, weights, init_splits_guesses):
        
        self.params_lst = params_lst
        self.weights = weights
        self.init_splits_guesses = init_splits_guesses
        
        # extra variable to store the optimal splits
        self._cv_preds = []
        self._test_preds = []
        self._splits = None

    def _ensemble(self, mode='cv'):
        
        if mode is 'cv':
            y_lst = self._cv_preds
        elif mode is 'test':
            y_lst = self._test_preds
        else:
            raise ValueError("mode should be either 'cv' or 'test'!!!")
            
        # Calculate ensemble results given predictions and weights
        combined = np.empty([y_lst[0].shape[0], len(y_lst)])
        for col, y in enumerate(y_lst):
            combined[:, col] = y
    
        return np.average(combined, axis=1, weights=self.weights)
        
    def _cv_train(self, X, y):
        
        num_models = len(self.params_lst)
        print "Cross validating {} models:".format(num_models)
        
        skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=52)
        
        for i, params in enumerate(self.params_lst):
            
            print "Model {} started;".format(i+1)
            plst, num_rounds = params[0], params[1]
            y_val = y.copy().astype(float)
            
            for train_idx, val_idx in skf:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                train_xgb = xgb.DMatrix(X_train, label=y_train)
                val_xgb = xgb.DMatrix(X_val)
                rgr = xgb.train(plst, train_xgb, num_rounds)
                y_val[val_idx] = rgr.predict(val_xgb)
            
            self._cv_preds.append(y_val)
            print "Model {} completed!".format(i+1)
        
        print ''
        
    def fit(self, X, y):
        
        self._cv_train(X, y)
        y_cv = self._ensemble(mode='cv')
        self._splits = splits_opt(y, y_cv, self.init_splits_guesses)
        
        print "Best splits is {}\n".format(self._splits)
        
        return self
        
    def predict(self, X, y, X_test, test_id, file_out):
        
        num_models = len(self.params_lst)
        print "Training {} models:".format(num_models)
        
        # train the models and predict for test
        for i, params in enumerate(self.params_lst):
            
            print "Model {} started;".format(i+1)
            plst, num_rounds = params[0], params[1]
            train_xgb = xgb.DMatrix(X, label=y)
            test_xgb = xgb.DMatrix(X_test)
            rgr = xgb.train(plst, train_xgb, num_rounds)
            self._test_preds.append(rgr.predict(test_xgb))
            print "Model {} completed!".format(i+1)
        
        # ensemble the test predicts
        y_test = self._ensemble(mode='test')
        
        # using the splits to map the raw predictions to ordinal numbers
        y_test_oridinal = [digitize(yt, self._splits) for yt in y_test]
        
        # save the results to file
        df_result = pd.DataFrame({'Id': test_id, 'Response': y_test_oridinal})
        df_result.set_index('Id', inplace=True)
        df_result.to_csv(file_out)
        print "\nPrediction for test set is done!"
    
    
def main():
    
    # read in the data
    df_train = pd.read_csv(FOLDER+FILE1)
    df_test = pd.read_csv(FOLDER+FILE2)
    
    # fill the null values
    df_train, df_test = fill_null(df_train, df_test)
    
    # prepare X and y
    y = df_train['Response'].values
    X = df_train.drop(['Id', 'Response'], axis=1).values

    # predict for test set
    test_id = df_test['Id'].astype(int).values
    X_test = df_test.drop(['Id'], axis=1).values

    # set up initial split guess values
    init_splits_guesses = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    
    # set up the parameters for the first model
    params1 = {'objective': 'reg:linear', 
              'eta': 0.005,
              'subsample': 0.7,
              'max_depth': 6,
              'min_child_weight': 1,
              'colsample_bytree': 0.7,
              'gamma': 5,
              'alpha': 0.5,
              'lambda': 1,
              'silent': 1
              }
    num_rounds1 = 6000
    
    # set up the parameters for the second model
    params2 = {'objective': 'reg:linear', 
              'eta': 0.005,
              'subsample': 0.7,
              'max_depth': 6,
              'min_child_weight': 1,
              'colsample_bytree': 0.7,
              'gamma': 0,
              'alpha': 0,
              'lambda': 1,
              'silent': 1
              }
    num_rounds2 = 6000
    
    # wrap up the parameters list
    params_lst = [[list(params1.items()), num_rounds1],
                  [list(params2.items()), num_rounds2]]
    
    # set up the weights for models
    weights = [2, 1]
    
    # initialize the model class, find the optimal splits and predict for test
    pm = PrudentialModel(params_lst, weights, init_splits_guesses)
    pm = pm.fit(X, y)
    file_out = FOLDER + 'submission.csv'
    pm.predict(X, y, X_test, test_id, file_out)
    

if __name__ == '__main__':

    main()
    print "--- {:.2f} seconds ---".format(time.time() - start_time)
