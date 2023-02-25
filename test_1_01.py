from os import getcwd
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.impute import KNNImputer

#Variations: (1) Model (sklearn, xgboost, keras) (2) NA handling technique (mode, kNN)  (3) No. of k neighbors for kNN imputer (4) Hyperparameter tuning

##Section 1: Introduction and loading the data
#Initializing files and directories
pd.set_option('display.max_rows',3000)

wd = getcwd()
x_train_file = wd + '\\offline\\X_train.csv'
y_train_file = wd + '\\offline\\y_train.csv'

x_test_file = wd + '\\online\\X_test.csv'
y_test_file = wd + '\\online\\y_test.csv'

x_pred_file = wd + '\\online\\X_test_submission.csv'

offline_pininfo_file = wd + '\\offline\\PinInfo.csv'
online_pininfo_file = wd + '\\online\\PinInfo.csv'

#Reading csv into dataframes
x_train_df = pd.read_csv(x_train_file)
y_train_df = pd.read_csv(y_train_file)

x_test_df = pd.read_csv(x_test_file)
y_test_df = pd.read_csv(y_test_file)

x_pred_df = pd.read_csv(x_pred_file)

offline_pininfo_df = pd.read_csv(offline_pininfo_file)
online_pininfo_df = pd.read_csv(online_pininfo_file)

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

label_df = pd.DataFrame(['1'] * len_x_train_df + ['2'] * len_x_test_df + ['3'] * len_x_pred_df,columns=['label']) #Labels: 1:Train, 2:Test, 3:Pred

#Combining all the X-values from x_train, x_test and x_pred to visualize the distribution
x_comb_df = pd.concat([x_train_df,x_test_df,x_pred_df],axis=0)
x_comb_df = x_comb_df.reset_index(drop=True) #The index of the original dataframes are dropped to merge with the label_df
x_comb_df = pd.concat([x_comb_df,label_df],axis=1)
#print(x_comb_df)

comb_pininfo_df = pd.concat([offline_pininfo_df,online_pininfo_df],axis=0)
#print(x_comb_df)

#Distribution of NAs by column
col_list = x_comb_df.columns
col_na_list = []
for i in col_list:
    col_na_list.append(x_comb_df[i].isna().sum())
col_na_df = pd.concat([pd.DataFrame(col_list,columns=['BLE']),pd.DataFrame(col_na_list,columns=['No. of NaNs'])],axis=1)

#print(x_train_df['CD4533FFC0E1'].values)
#print(x_train_df[x_train_df[]])
#df[df.columns[x]].values

#Visualizing the total no. of NAs by BLE
fix_na = px.pie(col_na_df,values='No. of NaNs',names='BLE',title='Combined No. of NaNs by BLE transmitters')
fix_na.show()

#Visualizing the data point distributions in a violin-boxplot
row_titles = ['Train','Test','Pred']
fig_subplot = make_subplots(rows=3,cols=17,row_titles=row_titles)
df_list = [x_train_df,x_test_df,x_pred_df]
for i,j in enumerate(df_list):
    for k,l in enumerate(j.columns):
        fig_subplot.append_trace(go.Violin(y=j[l].values,name=str(l),fillcolor='lightblue',line_color='grey'),row=i+1,col=k+1)
        #fig_subplot.append_trace(go.Violin(x=j.columns,y=j[j[l]],name=str(l)),row=i+1,col=1)
fig_subplot.update_xaxes(tickangle=20)
fig_subplot.layout.update(showlegend=False,title_text = 'Distribution of BLE transmitters signal strength by dataset')
fig_subplot.show()

#Visualizing the pin info xy-spatial distribution
#Comment: It looks quite uniform
fig_y = px.scatter(comb_pininfo_df,x='x',y='y')
fig_y.show()

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

x_train_df = x_comb_df[x_comb_df['label']==1]
x_test_df = x_comb_df[x_comb_df['label']==2]
x_pred_df = x_comb_df[x_comb_df['label']==3]
