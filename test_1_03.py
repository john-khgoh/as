from os import getcwd
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.multioutput import MultiOutputRegressor
#from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

#Variations: (1) Model (sklearn, xgboost, keras) (2a) NA handling technique (mode, kNN)  (2b) No. of k neighbors for kNN imputer (3) Hyperparameter tuning

##Section 1: Introduction and loading the data
#Note: Explain hypothesis & assumptions
#E.g. An assumption is that there's a moderate to strong correlation between the observed signal strengths and the position of the BLE receivers relative to the BLE transmitters.
#Otherwise, performing ML to predict location of BLE receivers would not yield any meaningful results (explain)
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
#print(x_train_df.shape)
#print(x_test_df.shape)
#print(x_pred_df.shape)

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
#Counting the percentage of NaN for x_train
x_train_df_na = x_train_df.isna().sum().sum()
x_train_df_size = x_train_df.shape[0] * x_train_df.shape[1]
x_train_na_pct = 100 * x_train_df_na / x_train_df_size
#print("%.2f" %x_train_na_pct)

#Counting the percentage of NaN for x_test
x_test_df_na = x_test_df.isna().sum().sum()
x_test_df_size = x_test_df.shape[0] * x_test_df.shape[1]
x_test_na_pct = 100 * x_test_df_na / x_test_df_size
#print("%.2f" %x_test_na_pct)

#Counting the percentage of NaN for x_test_submission
#Counting the percentage of NaN for x_test
x_pred_df_na = x_pred_df.isna().sum().sum()
x_pred_df_size = x_pred_df.shape[0] * x_pred_df.shape[1]
x_pred_na_pct = 100 * x_pred_df_na / x_pred_df_size
#print("%.2f" %x_pred_na_pct)

#Comment: There's a significant large no. of NaNs in the dataset, especially in x_test_submission
#Comment: The no. of NaNs by BLE is as shown by the previous pie chart
#Comment: A data imputation technique is required. It's possible to substitute it with: (1)Zeros (2)Mean (3)Mode (4)k-Nearest Neighbors
imputer = KNNImputer(n_neighbors=5,weights='distance')
imputer.fit(x_comb_df)
x_comb = imputer.transform(x_comb_df)
x_comb_df = pd.DataFrame(x_comb,columns=col_list)
#print(x_comb_df)

#Splitting the imputed datasets back into the original train,test and pred datasets
x_train_df = x_comb_df[x_comb_df['dataset']==1]
x_test_df = x_comb_df[x_comb_df['dataset']==2]
x_pred_df = x_comb_df[x_comb_df['dataset']==3]

#print(x_train_df)

##Section 4: Model training and prediction
#Dropping the dataset labels because it's not required for prediction
x_train_df = x_train_df.drop(columns=['dataset'])
x_test_df = x_test_df.drop(columns=['dataset'])
x_pred_df = x_pred_df.drop(columns=['dataset'])

#Merge y-dataframes with PinInfo and dropping the PinId
y_train_df.columns = y_test_df.columns = ['pinId']

y_train_df = pd.merge(y_train_df,offline_pininfo_df)
y_test_df = pd.merge(y_test_df,online_pininfo_df)

y_train_df = y_train_df.drop(columns=['pinId'])
y_test_df = y_test_df.drop(columns=['pinId'])

#Training the model
x_train_arr = np.array(x_train_df)
y_train_arr = np.array(y_train_df)
x_test_arr = np.array(x_test_df)  

estimator = XGBRegressor(objective = 'reg:squarederror')
model = MultiOutputRegressor(estimator = estimator,n_jobs = -1).fit(x_train_arr, y_train_arr)

#Predicting value of yhat_test_df
yhat_test_arr = model.predict(x_test_arr)
yhat_test_df = pd.DataFrame(yhat_test_arr,columns=['x','y'])
print(yhat_test_arr)
print(y_test_df)

