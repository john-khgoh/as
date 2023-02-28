#Keras

from os import getcwd
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.initializers import RandomNormal
#from tensorflow.python.keras.optimizer_v2 import Adam

#Variations: (1) Model (sklearn, xgboost, keras) (2a) NA handling technique (mode, kNN)  (2b) No. of k neighbors for kNN imputer (3) Hyperparameter tuning (4) Cross validation

##Section 1: Introduction and loading the data
#Note: Explain hypothesis & assumptions
#E.g. An assumption is that there's a moderate to strong correlation between the observed signal strengths and the position of the BLE receivers relative to the BLE transmitters.
#Otherwise, performing ML to predict location of BLE receivers would not yield any meaningful results (explain)
#Another assumption is that a large number of NaNs does not significantly affect the model's ability to predict. 
#We will later see that the percentage of NaNs is higher than the threshold of missing data (~10%)
#Another assumption is that there's enough training data to create a robust model. 1500+ data points for training is not always sufficient to create a robust model. 
#Another assumption is that external signal interference should not affect the readings
#Initializing files and directories
pd.set_option('display.max_rows',3000)
pd.set_option('display.max_columns',100)

wd = getcwd()
x_train_file = wd + '\\offline\\X_train.csv'
y_train_file = wd + '\\offline\\y_train.csv'
x_test_file = wd + '\\online\\X_test.csv'
y_test_file = wd + '\\online\\y_test.csv'
x_pred_file = wd + '\\online\\X_test_submission.csv'
output_file = wd + '\\submissions.csv'

offline_pininfo_file = wd + '\\offline\\PinInfo.csv' #train
online_pininfo_file = wd + '\\online\\PinInfo.csv' #test

#Reading csv into dataframes
x_train_df = pd.read_csv(x_train_file)
y_train_df = pd.read_csv(y_train_file)
x_test_df = pd.read_csv(x_test_file)
y_test_df = pd.read_csv(y_test_file)
x_pred_df = pd.read_csv(x_pred_file)

offline_pininfo_df = pd.read_csv(offline_pininfo_file) #train
online_pininfo_df = pd.read_csv(online_pininfo_file) #test

##Section 2: Data exploration
#Getting the dataframe dimensions
print(x_train_df.shape)
print(x_test_df.shape)
print(x_pred_df.shape)

#Description of the data distribution for x_train and x_test
#print(x_train_df.describe())
#print(x_test_df.describe())

#Adding a column to label the original dataset
len_x_train_df = len(x_train_df)
len_x_test_df = len(x_test_df)
len_x_pred_df = len(x_pred_df)

dataset_label_df = pd.DataFrame(['1'] * len_x_train_df + ['2'] * len_x_test_df + ['3'] * len_x_pred_df,columns=['dataset']) #Labels: 1:Train, 2:Test, 3:Pred

#Combining all the X-values from x_train, x_test and x_pred to visualize the distribution
x_comb_df = pd.concat([x_train_df,x_test_df,x_pred_df],axis=0)
x_comb_df = x_comb_df.reset_index(drop=True) #The index of the original dataframes are dropped to merge with the dataset_label_df
x_comb_df = pd.concat([x_comb_df,dataset_label_df],axis=1)
#print(x_comb_df)

onoff_pin_label_df = pd.DataFrame(['Off'] * len(offline_pininfo_df) + ['On'] * len(online_pininfo_df),columns=['status'])
comb_pininfo_df = pd.concat([offline_pininfo_df,online_pininfo_df],axis=0)
comb_pininfo_df = comb_pininfo_df.reset_index(drop=True)
comb_pininfo_df = pd.concat([comb_pininfo_df,onoff_pin_label_df],axis=1)
#print(comb_pininfo_df)

#Distribution of NAs by column
col_list = x_comb_df.columns
col_na_list = []
for i in col_list:
    col_na_list.append(x_comb_df[i].isna().sum())
col_na_df = pd.concat([pd.DataFrame(col_list,columns=['BLE']),pd.DataFrame(col_na_list,columns=['No. of NaNs'])],axis=1)

#Visualizing the total no. of NAs by BLE
fix_na = px.pie(col_na_df,values='No. of NaNs',names='BLE',title='Combined No. of NaNs by BLE receivers')
#fix_na.show()

#Visualizing the data point distributions in a violin-boxplot
row_titles = ['Train','Test','Pred']
fig_subplot = make_subplots(rows=3,cols=17,row_titles=row_titles)
df_list = [x_train_df,x_test_df,x_pred_df]
for i,j in enumerate(df_list):
    for k,l in enumerate(j.columns):
        fig_subplot.append_trace(go.Violin(y=j[l].values,name=str(l),fillcolor='lightblue',line_color='grey'),row=i+1,col=k+1)
        #fig_subplot.append_trace(go.Violin(x=j.columns,y=j[j[l]],name=str(l)),row=i+1,col=1)
fig_subplot.update_xaxes(tickangle=20)
fig_subplot.layout.update(showlegend=False,title_text = 'Distribution of BLE Receivers Signal Strength by Dataset')
#fig_subplot.show()

#Visualizing the pin info xy-spatial distribution
#Comment: It looks quite uniform
fig_y = px.scatter(comb_pininfo_df,x='x',y='y',color='status',color_discrete_sequence=['red','green'],title='Pin Info XY-spatial Distribution')
#fig_y.show()

##Section 3: NA handling
#Counting the percentage of NaN for a dataframe
#Comment: There's a significant large no. of NaNs in the dataset, especially in x_test_submission
#Comment: The no. of NaNs by BLE is as shown by the previous pie chart
#Comment: A data imputation technique is required. It's possible to substitute it with: (1)Zeros (2)Mean (3)Mode (4)k-Nearest Neighbors
def count_na_pct(df):
    df_na = df.isna().sum().sum()
    df_size = df.shape[0] * df.shape[1]
    df_na_pct = 100 * df_na / df_size
    return df_na_pct

x_train_na_pct = count_na_pct(x_train_df)
x_test_na_pct = count_na_pct(x_test_df)
x_pred_na_pct = count_na_pct(x_pred_df)
print("%.2f %.2f %.2f" %(x_train_na_pct,x_test_na_pct,x_pred_na_pct))

imputer = KNNImputer(n_neighbors=14)
imputer.fit(x_comb_df)
x_comb = imputer.transform(x_comb_df)
x_comb_df = pd.DataFrame(x_comb,columns=col_list)
#print(x_comb_df)

#Splitting the imputed datasets back into the original train,test and pred datasets
x_train_df = x_comb_df[x_comb_df['dataset']==1]
x_test_df = x_comb_df[x_comb_df['dataset']==2]
x_pred_df = x_comb_df[x_comb_df['dataset']==3]

##Section 4: Model training and prediction
#Dropping the dataset labels because it's not required for prediction
x_train_df = x_train_df.drop(columns=['dataset'])
x_test_df = x_test_df.drop(columns=['dataset'])
x_pred_df = x_pred_df.drop(columns=['dataset'])
x_train_test_df = x_comb_df[x_comb_df['dataset']!=3]
x_train_test_df = x_train_test_df.drop(columns=['dataset'])

#Merge y-dataframes with PinInfo and dropping the PinId
y_train_df.columns = y_test_df.columns = ['pinId']

y_train_df = pd.merge(y_train_df,offline_pininfo_df)
y_test_df = pd.merge(y_test_df,online_pininfo_df)

y_train_df = y_train_df.drop(columns=['pinId'])
y_test_df = y_test_df.drop(columns=['pinId'])

y_train_test_df = pd.concat([y_train_df,y_test_df],axis=0)
y_train_test_df = y_train_test_df.reset_index(drop=True)

#Training the model
x_train_arr = np.array(x_train_df)
x_test_arr = np.array(x_test_df)
x_pred_arr = np.array(x_pred_df)
x_train_test_arr = np.array(x_train_test_df)

y_train_arr = np.array(y_train_df)
y_test_arr = np.array(y_test_df)
y_train_test_arr = np.array(y_train_test_df)

#estimator = lgb.LGBMRegressor()
#estimator = RandomForestRegressor(n_estimators=120)
#estimator = XGBRegressor(n_estimators=100,eta=0.25,max_depth=6)
#Use a Keras sequential model

seed = [20,40,60]
n_inputs = x_train_df.shape[1]
n_outputs = y_train_df.shape[1]
seq = Sequential()
seq.add(Dense(20, input_dim=n_inputs,kernel_initializer=RandomNormal(seed=seed[0]),activation='relu'))
seq.add(Dense(10,kernel_initializer=RandomNormal(seed=seed[1]),activation='relu'))
seq.add(Dense(4,kernel_initializer=RandomNormal(seed=seed[2]),activation='relu'))
seq.add(Dense(n_outputs,activation='relu'))
seq.compile(loss='mae',optimizer='Adam')
seq.fit(x_train_arr, y_train_arr,verbose=1,epochs=250)
yhat_test_arr = seq.predict(x_test_arr)

#model = MultiOutputRegressor(estimator = seq).fit(x_train_arr, y_train_arr)

#Checking the accuracy of predictions
def error_calculation(y,yhat):
    summed_x_error = 0.0
    summed_y_error = 0.0
    summed_xy_error = 0.0
    len_y = len(y)
    for i in range(len_y):
        x_diff = abs(y['x'][i] - yhat['x'][i]) #Difference of x-coordinates between y_hat and y
        y_diff = abs(y['y'][i] - yhat['y'][i]) #Difference of y-coordinates between y_hat and y
        summed_x_error += x_diff
        summed_y_error += y_diff
        summed_xy_error += ((x_diff + y_diff) / 2.0) #Getting the average between the two
    mean_x_error = summed_x_error / len_y
    mean_y_error = summed_y_error / len_y
    mean_xy_error = summed_xy_error / len_y
    return mean_x_error,mean_y_error,mean_xy_error

#Predicting value of yhat_test_df
#yhat_test_arr = model.predict(x_test_arr)
yhat_test_df = pd.DataFrame(yhat_test_arr,columns=['x','y'])
mean_x_error,mean_y_error,mean_xy_error = error_calculation(y_test_df,yhat_test_df)
print("%.3f,%.3f,%.3f" %(mean_x_error,mean_y_error,mean_xy_error))

'''
#Predicting the values for x_pred using the combined x_train and x_test data
model = MultiOutputRegressor(estimator = estimator,n_jobs = -1).fit(x_train_test_arr, y_train_test_arr)
yhat_pred_arr = model.predict(x_pred_arr)
yhat_pred_df = pd.DataFrame(yhat_pred_arr,columns=['x','y'])
yhat_pred_df.to_csv(output_file,index=False)

#What next:
#Write the report
#Use optuna to select the hyperparameters
#Use k-fold cross validation to select the model (optional)
'''