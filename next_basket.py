"""The model for the next basket predictions.
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from numba import jit

from datetime import date as dt
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from xgboost import XGBClassifier

dataFolder = 'Data\\'

def read_file(data_folder: str) -> pd.DataFrame:
	"""Read csv files from the data folder.
    
    Args:
        data_folder: the name of the folder which stores all the csv files.
        
    Returns:
        Transactions in pd.DataFrame.
	"""

	dataFiles = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
	container = []
	for name in dataFiles:
		df = pd.read_csv(name, encoding="ISO-8859-1", sep=';')
		container.append(df)
	df = pd.concat(container)
	df.columns = ['date_time', 'customer_id', 'age', 'area', 'subclass', 'product_id', 
	'amount', 'asset', 'sale_price']

	df['profit'] = df['sale_price'] - df['asset']
	df['date_time'] = df['date_time'].apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))
	df['weekday'] = df['date_time'].apply(lambda x: x.weekday()) 
	df['month'] = df.date_time.astype(str).apply(lambda x:x[5:7])

	return df


@jit
def add_transaction_id(df: pd.DataFrame) -> pd.DataFrame:
	"""Create transaction id for each product purchased (row in df). Products in the same basket have the same transaction id.

	Args:
        df: Dataframe which includes all the product purchased.
        
    Returns:
        df: Add transaction id to the input df.

	"""
	df['trans_id'] =  df['customer_id'].astype(str) + df['date_time'].astype(str) # assume each customer only make at most one transaction everyday
	df = df.sort_values(['trans_id']) # sort before finding the products in the same basket
	cust_id = df.customer_id.values[0]
	trans_buf_id = df.trans_id.values[0]
	trans = 0
	trans_id_list = []
	for i in tqdm(range(df.shape[0])):
		if df.customer_id.values[i] == cust_id:
			if df.trans_id.values[i] == trans_buf_id:
				trans_id_list.append(trans)
			else:
				trans += 1
				trans_buf_id = df.trans_id.values[i]
				trans_id_list.append(trans)
		else:
			cust_id = df.customer_id.values[i]
			trans_buf_id = df.trans_id.values[i]
			trans = 0
			trans_id_list.append(trans)
	df['trans_id'] = trans_id_list

	return df


def create_feature_for_one_customer(df_cust: pd.DataFrame, features: list, targets: list, train_test: list):
	"""Create modeling features for each cutomer. Uses each transaction's previous two transactions information.

	Args:
        df_cust: Dataframe which includes all the transactions made by a single customer.
        features: a list of lists;
        			day of the week of the transactions, difference between the current transaction and the previous two transactions in days,
					amount of all products, amount of the certain product, item prices, month of the transactions, product information, 
					customer information. 
        targets: a list of 0, 1. 0 indicates a product is not purchased in the current transaction.
        		1 indicates a product is purchased in the current transaction.
        train_test: a list of 0, 1. 1 indicates the corresponding feature will be usesd for training.
        			0 indicates the corresponding feature will be used for tessting.

	"""

	if df_cust.trans_id.max() < 4: # exclude customers with less than 4 transactions
		return

	for i in range(2, df_cust.trans_id.max() + 1): # start from the third transaction of each cutomer
		if i == df_cust.trans_id.max(): # if it is the last order
			train_test = 0
		else:
			train_test = 1
		df_prev = df_cust.loc[df_cust.trans_id.isin([i - 1, i - 2])].copy()
		df_prev_1 = df_prev.loc[df_prev.trans_id == i - 1]
		df_prev_2 = df_prev.loc[df_prev.trans_id == i - 2]
		curr_product = set(df_cust.loc[df_cust.trans_id == i,'product_id'])
		curr_date = df_cust.loc[df_cust.trans_id == i,'date_time'].values[0]
		feature = []
		last_weekday = df_prev_1['weekday'].values[0]
		last_weekday_2 = df_prev_2['weekday'].values[0]
		this_weekday = df_cust.loc[df_cust.trans_id == i, 'weekday'].values[0]
		diff_last_day = (curr_date - df_prev_1['date_time'].values[0]) / np.timedelta64(1, 'D')
		diff_last_day_2 = (curr_date - df_prev_2['date_time'].values[0]) / np.timedelta64(1, 'D')
		last_all_amount = df_prev_1['amount'].sum()
		last_all_amount_2 = df_prev_2['amount'].sum()
		cust_area = df_cust['area'].values[0]
		cust_age = df_cust['age'].values[0]
		last_month = str(df_prev_1.date_time.values[0])[5:7]
		last_month_2 = str(df_prev_2.date_time.values[0])[5:7]
		cust_id = df_cust['customer_id'].values[0]
		for prod_id in df_prev.product_id.unique():
			if prod_id in curr_product:
				targets.append(1)
			else:
				targets.append(0)
			dfdf_last = df_prev_1.loc[(df_prev_1.product_id == prod_id)].copy()
			dfdf_last_2 = df_prev_2.loc[(df_prev_2.product_id == prod_id)].copy()
			last_amount = 0
			last_amount_2 = 0
			last_price = 0
			last_price_2 = 0

			try:
				last_amount = dfdf_last['amount'].values[0]
				last_price = dfdf_last['sale_price'].values[0] / last_amount
			except:
				None
        
			try:
				last_amount_2 = dfdf_last_2['amount'].values[0]
				last_price_2 = dfdf_last_2['sale_price'].values[0] / last_amount_2
			except:
				None
        
			subcls = df_prev.loc[df_prev.product_id == prod_id,'subclass'].values[0]
			feature = [last_weekday, last_weekday_2, this_weekday, diff_last_day, diff_last_day_2, last_all_amount, last_all_amount_2,
						last_amount, last_amount_2, last_price, last_price_2, last_month, last_month_2, int(subcls), int(prod_id), cust_area, 
						cust_age, cust_id, train_test]
			features.append(feature)

	return


@jit
def create_feature(df: pd.DataFrame) -> pd.DataFrame:
	"""Create features and targets from the transaction dataframe.
	
	Args:
        df: Dataframe which includes all transactions.
        
    Returns:
        df_feature_target: a dataframe includes features and targets.
        					df_feature_target['Y'] are the targets.

	"""

	cust_id = df.customer_id.values[0]
	start = 0
	features = [] # initial feature sets
	targets = [] # initial targets
	train_test = [] # initial train_test indicator
	for i in tqdm(range(df.shape[0])): # find transactions of each customer
		if df['customer_id'].values[i] == cust_id:
			continue
		df_cust = df.iloc[start:i,:].copy()
		create_feature_for_one_customer(df_cust, features, targets, train_test)
		cust_id = df.customer_id.values[i]
		start = i

	df_feature_target = pd.DataFrame(features)
	df_feature_target['Y'] = targets

	return df_feature_target


def train_test_split(df_feaure_target: pd.DataFrame) -> (np.array, np.array, np.array, np.array):
	"""Onehot encoding plus train test split.

	Args:
        df_feaure_target: Dataframe with raw features and targets.
        
    Returns:
        X_train:
        X_test:
        y_train:
        y_test:

	"""
	# onehot encoding customer area information
	label_encoder = LabelEncoder()
	lb_f = label_encoder.fit_transform(df_feaure_target[15]).reshape(-1,1)
	one_hot = OneHotEncoder(sparse=False)
	oh_f = one_hot.fit_transform(lb_f)

	# onehot encoding customer age information
	label_encoder = LabelEncoder()
	lb_f = label_encoder.fit_transform(df_feaure_target[16]).reshape(-1,1)
	one_hot = OneHotEncoder(sparse=False)
	oh_f_1 = one_hot.fit_transform(lb_f)

	bools = df_feaure_target[18] == 1 # identify training set
	del df_feaure_target[15], df_feaure_target[16], df_feaure_target[18]

	X = df_feaure_target.iloc[:,:-1].values
	X = np.concatenate((X, oh_f), axis=1)
	X = np.concatenate((X, oh_f_1), axis=1) 

	Y = df_feaure_target.iloc[:,-1].values

	X_train, X_test, y_train, y_test = X[bools, :], X[~bools, :], Y[bools], Y[~bools]

	return X_train, X_test, y_train, y_test



def plt_auc(model, X_test, Y_test, test_start):
	"""Plot ROC area and return roc score.
	"""
	probs = model.predict_proba(X_test)

	preds = probs[:,1]
	fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
	roc_auc = metrics.auc(fpr, tpr)
	#y_pred = xgb1.predict(X[train_len:])

	plt.title(str(test_start) + 'Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()
	return roc_auc



if __name__ == "__main__":

	print('Reading files ...\n')
	df = read_file(dataFolder)

	print('Adding transaction numbers ...\n')
	df = add_transaction_id(df)


	# create features and targets
	print('Creating features ...\n')
	df_feature_target = create_feature(df)


	X_train, X_test, y_train, y_test = train_test_split(df_feature_target)

	num_of_eval = int(len(y_test) * 0.5)
	model = XGBClassifier(n_estimators=1000, max_depth=7, colsample_bytree=0.7, nthread=-1)
	eval_set = [(X_test[:num_of_eval, :], y_test[:num_of_eval])]

	# I use early stopping here to prevent model from overfitting to the training set.
	# Cross validation is a method to avoid overfitting when tunning the model parameters.
	model.fit(X_train, Y_train, early_stopping_rounds=100, eval_metric="auc",eval_set=eval_set, verbose=True)

	# record the number of trees before the model overfitting to the training set and retrain the model
	num_tree = 200 
	model = XGBClassifier(n_estimators=num_tree, max_depth=7, colsample_bytree=0.7, nthread=-1)
	model.fit(X_train, Y_train)

	# plot roc curve
	roc_auc = plt_auc(model,X_test[num_of_eval:,:],y_test[num_of_eval:],'test')

	
	y_pred = model.predict_proba(X_test[num_of_eval:,:])[:,1]
	y_pred = [1 if i > 0.18 else 0 for i in y_pred] # choose difference threshold to meet different requirements
	print("Recall: ", recall_score(y_test[num_of_eval:],y_pred)) 
	print("Precision: ", precision_score(y_test[num_of_eval:],y_pred))
	print("Accuracy: ", accuracy_score(y_test[num_of_eval:],y_pred))
	prin("F1: ", f1_score(y_test[num_of_eval:],y_pred))
