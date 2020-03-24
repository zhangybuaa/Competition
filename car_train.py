import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import  display,clear_output
import time

warnings.filterwarnings('ignore')


## 模型预测的
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

Train_data = pd.read_csv('./data/used_car_train_20200313.csv', sep=' ')
TestA_data = pd.read_csv('./data/used_car_testA_20200313.csv', sep=' ')

# print('Train data shape:',Train_data.shape)
# print('TestA data shape:',TestA_data.shape)

Train_data.head()

numerical_cols = Train_data.select_dtypes(exclude = 'object').columns

categorical_cols = Train_data.select_dtypes(include = 'object').columns

## 选择特征列
feature_cols = [col for col in numerical_cols if col not in ['SaleID','name','regDate','creatDate','price','model','brand','regionCode','seller']]
feature_cols = [col for col in feature_cols if 'Type' not in col]

## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
Y_data = Train_data['price']

X_test  = TestA_data[feature_cols]

X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)

def Sta_inf(data):
    print('_min',np.min(data))
    print('_max:',np.max(data))
    print('_mean',np.mean(data))
    print('_ptp',np.ptp(data))
    print('_std',np.std(data))
    print('_var',np.var(data))

## xgb-Model
# xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8, \
#                        colsample_bytree=0.9, max_depth=7)  # ,objective ='reg:squarederror'

# scores_train = []
# scores = []
#
# # 5折交叉验证方式
# sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# i = 1
# for train_ind, val_ind in sk.split(X_data, Y_data):
#
#     train_x = X_data.iloc[train_ind].values
#     train_y = Y_data.iloc[train_ind]
#     val_x = X_data.iloc[val_ind].values
#     val_y = Y_data.iloc[val_ind]
#
#     xgr.fit(train_x, train_y)
#     pred_train_xgb = xgr.predict(train_x)
#     pred_xgb = xgr.predict(val_x)
#
#     score_train = mean_absolute_error(train_y, pred_train_xgb)
#     scores_train.append(score_train)
#     score = mean_absolute_error(val_y, pred_xgb)
#     scores.append(score)
#     print('fold %d score_train: %f val_score :%f' % (i, score_train, score))
#     i += 1
#
# print('Train mae:', np.mean(score_train))
# print('Val mae', np.mean(scores))

# 定义xgb和lgb模型函数
def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=7) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model

def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127,n_estimators = 150)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm

## Split data with val
x_train,x_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.3)

#用lgb模型训练
print('Train lgb...')
model_lgb = build_model_lgb(x_train,y_train)
val_lgb = model_lgb.predict(x_val)
MAE_lgb = mean_absolute_error(y_val,val_lgb)
print('MAE of val with lgb:',MAE_lgb)

print('Predict lgb...')
model_lgb_pre = build_model_lgb(X_data,Y_data)
Ytest_lgb = model_lgb_pre.predict(X_test)
print('Sta of Predict lgb:')
Sta_inf(Ytest_lgb)

#用xgb模型训练
print('Train xgb...')
model_xgb = build_model_xgb(x_train,y_train)
val_xgb = model_xgb.predict(x_val)
MAE_xgb = mean_absolute_error(y_val,val_xgb)
print('MAE of val with xgb:',MAE_xgb)

print('Predict xgb...')
model_xgb_pre = build_model_xgb(X_data,Y_data)
Ytest_xgb = model_xgb_pre.predict(X_test)
print('Sta of Predict xgb:')
Sta_inf(Ytest_xgb)

## 模型融合，采取了简单的加权融合的方式
val_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*val_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*val_xgb
val_Weighted[val_Weighted<0]=10 # 由于我们发现预测的最小值有负数，而真实情况下，price为负是不存在的，由此我们进行对应的后修正
print('MAE of val with Weighted ensemble:',mean_absolute_error(y_val,val_Weighted))


Y_test = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*Ytest_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*Ytest_xgb

## 查看预测值的统计进行
plt.hist(Y_test)
plt.show()
plt.close()


#输出结果
print('生成结果文件')
sub = pd.DataFrame()
sub['SaleID'] = TestA_data.SaleID
sub['price'] = Y_test
sub.to_csv('./sub_Weighted.csv',index=False)

