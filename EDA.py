import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno



Train_data = pd.read_csv('./data/used_car_train_20200313.csv', sep=' ')
TestA_data = pd.read_csv('./data/used_car_testA_20200313.csv', sep=' ')


# print(Train_data.head())
# print(Train_data.describe())
# print(Train_data.info())
# print(Train_data.isnull())
#可视化1
# missing = Train_data.isnull().sum()
# missing = missing[missing > 0]
# missing.sort_values(inplace=True)
# missing.plot.bar()
# plt.show()

# msno.matrix(Train_data.sample(250))
# msno.bar(Train_data)
# plt.show()

# print(Train_data["seller"].value_counts())
# print(Train_data["price"].value_counts())
#
# import scipy.stats as st
# y = Train_data['price']
# plt.figure(1); plt.title('Johnson SU')
# sns.distplot(y, kde=False, fit=st.johnsonsu)
# plt.figure(2); plt.title('Normal')
# sns.distplot(y, kde=False, fit=st.norm)
# plt.figure(3); plt.title('Log Normal')
# sns.distplot(y, kde=False, fit=st.lognorm)
# plt.show()

plt.hist(Train_data['price'])
plt.show()