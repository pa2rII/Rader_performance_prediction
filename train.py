import pandas as pd
import random
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import LassoCV , ElasticNetCV , RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.cross_decomposition import PLSRegression as  PLS
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
import seaborn as sns
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

PATH = "./data"

train_df = pd.read_csv(f'{PATH}/train.csv')
train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

# for col in train_x:
#   print(col,len(train_x[col].unique()))


cv = ShuffleSplit(n_splits=5 , test_size=0.3, random_state=42)
pipe_linear = Pipeline([('scl', StandardScaler()),
                        ('poly', PolynomialFeatures()),
                        ('fit', LinearRegression())])
pipe_lasso = Pipeline([('scl', StandardScaler()),
                       ('poly', PolynomialFeatures()),
                       ('fit', Lasso(random_state = 42))])
pipe_ridge = Pipeline([('scl', StandardScaler()),
                       ('poly', PolynomialFeatures()),
                       ('fit', Ridge(random_state = 42))])
pipe_pca = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA()),
                     ('fit', Ridge(random_state = 42))])
pipe_pls = Pipeline([('scl', StandardScaler()),
                     ('fit', PLS())])
pipe_gbr = Pipeline([('scl', StandardScaler()),
                     ('fit', GBR())])
pipe_rfr = Pipeline([('scl', StandardScaler()),
                     ('fit', RFR())])
pipe_svr = Pipeline([('scl', StandardScaler()),
                     ('fit', SVR())])
pipe_KR = Pipeline([('scl', StandardScaler()),
                    ('fit', KernelRidge())])
pipes = [
    pipe_linear , pipe_lasso ,  pipe_pca ,
    pipe_ridge , pipe_pls , pipe_gbr ,
    pipe_rfr , pipe_svr , pipe_KR
]
pipes_label = [
    'linear', 'lasso', 'pca', 'ridge',
    'pls', 'gbr', 'rfr', 'svr', 'KR'
]

grid_params_linear = [{
    "poly__degree" : np.arange(1,3),
    "fit__fit_intercept" : [True, False],
}]
grid_params_lasso = [{
    "poly__degree" : np.arange(1,3),
    "fit__tol" : np.logspace(-5,0,10) ,
    "fit__alpha" : np.logspace(-5,1,10) ,
                     }]
grid_params_pca = [{
    "pca__n_components" : np.arange(2,8)
}]
grid_params_ridge = [{
    "poly__degree" : np.arange(1,3),
    "fit__alpha" : np.linspace(2,5,10) ,
    "fit__solver" : [ "cholesky","lsqr","sparse_cg"] ,
    "fit__tol" : np.logspace(-5,0,10) ,
                     }]
grid_params_pls = [{
    "fit__n_components" : np.arange(2,8)
}]
min_samples_split_range = [0.5, 0.7 , 0.9]

grid_params_gbr =[{
    "fit__max_features" : ["sqrt","log2"] ,
    "fit__loss" : ["ls","lad","huber","quantile"] ,
    "fit__max_depth" : [5,6,7,8] ,
    "fit__min_samples_split" : min_samples_split_range ,
}]
grid_params_rfr =[{
    "fit__max_features" : ["sqrt","log2"] ,
    "fit__max_depth" : [5,6,7,8] ,
    "fit__min_samples_split" : min_samples_split_range ,
}]
grid_params_svr =[{
    "fit__kernel" : ["rbf", "linear"] ,
    "fit__degree" : [2, 3, 5] ,
    "fit__gamma" : np.logspace(-5,1,10) ,
}]
grid_params_KR =[{
    "fit__kernel" : ["rbf","linear"] ,
    "fit__gamma" : np.logspace(-5,1,10) ,
}]
params = [
    grid_params_linear , grid_params_lasso , grid_params_pca,
    grid_params_ridge , grid_params_pls , grid_params_gbr ,
    grid_params_rfr , grid_params_svr , grid_params_KR
]

pipes_dict = {}
for pipe, label in zip(pipes, pipes_label):
    _pipes = []
    for _ in range(14):
        _pipes.append(pipe)

    pipes_dict[label] = _pipes

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(14): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt.iloc[:,idx], preds.iloc[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt.iloc[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:7]) + 1.0 * np.sum(all_nrmse[7:13])
    return score


scores_dict = {}
preds_dict = {}
test_pred = {}
test_x = pd.read_csv(f'{PATH}/test.csv').drop(columns=['ID'])

cv = ShuffleSplit(n_splits=5 , test_size=0.3, random_state=42)

for pipe_key, pipelines in pipes_dict.items():
    scores = []
    preds = []
    preds_test = []
    print('-'*10,f'{pipe_key} pipeline training','-'*10)

    for i, (param, pipe) in tqdm(enumerate(zip(params, pipelines))):
        train_y_1d = train_y[train_y.columns[i]]
        valid_y_1d = valid_y[valid_y.columns[i]]

        search = GridSearchCV(pipe, param, scoring='neg_mean_squared_error', cv=cv, n_jobs=20, verbose=1)
        search.fit(train_x, train_y_1d)
        pred = search.predict(valid_x).reshape(-1)
        pred_test = search.predict(test_x).reshape(-1)

        preds.append(pred)
        preds_test.append(pred_test)

    preds_df = pd.DataFrame(preds).T
    preds_test_df = pd.DataFrame(preds_test).T
    print(f' nrmse : {lg_nrmse(valid_y, preds_df)}')

    submit = pd.read_csv(f'{PATH}/sample_submission.csv')
    for idx, col in enumerate(submit.columns):
        if col == 'ID':
            continue
        submit[col] = preds_test_df.iloc[:, idx - 1]
    print('Done.')
    submit.to_csv(f'./{pipe_key}_submit_GridSearch.csv', index=False)

