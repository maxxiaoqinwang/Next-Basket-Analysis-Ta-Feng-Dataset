"""A simple recommendation model based on collaborative filtering.
The performance is evaluation by randomly removing 1000 scores from the table.
And compare the estimation of these scores with the actual ones.
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

from next_basket import read_file, add_transaction_id

from datetime import date as dt
from datetime import datetime

from numpy import mat
import random
import svdRec

dataFolder = 'Data\\'

if __name__ == "__main__":

	print('Reading files ...\n')
	df = read_file(dataFolder)

	print('Adding transaction numbers ...\n')
	df = add_transaction_id(df)

	# the number of time that a customer purchassed a certain product
	df_cust_row = df.groupby(['customer_id','product_id'],as_index = False).agg({'trans_id':pd.Series.nunique})
	# Number of transactions made by each customer
	df_cust_num_trans = df.groupby(['customer_id'],as_index = False).agg({'trans_id':pd.Series.nunique})
	df_cust_num_trans.columns = ['customer_id','num_trans']
	# create a dataframe with columns: customer id, product id, number of times this product has been purchased, number of trans made by this customer
	df_rs = pd.merge(df_cust_row, df_cust_num_trans, on = 'customer_id', how = 'left')
	# compute score
	df_rs['score'] = df_rs['trans_id'] / df_rs['num_trans']

	# initialise the dataframe to store all the score information
	df_rs_sys = pd.DataFrame(np.zeros((df_rs['customer_id'].nunique(), df_rs.product_id.nunique())))
	# each column represesnts a product
	df_rs_sys.columns = list(df_rs.product_id.unique())
	# each row represents a customer
	df_rs_sys.index = list(df_rs['customer_id'].unique())

	# efficiently put the scores into df_rs_sys
	cust_id = df_rs.customer_id.values[0]
	start = 0
	print('Creating recsys table ...\n')
	for i in tqdm(range(df_rs.shape[0])):
		if df_rs.customer_id.values[i] == cust_id:
			continue
		df_buf = df_rs.iloc[start:i,:].copy()
		for j in range(df_buf.shape[0]):
			prod_id = df_buf['product_id'].values[j]
			score = df_buf['score'].values[j]
			df_rs_sys.loc[df_rs_sys.index == cust_id,prod_id] = score
		start = i
		cust_id = df_rs.customer_id.values[i] 


	# randomly remove 1000 scores from the table(df_rs_sys)
	c_level = random.sample(range(32266), 1000) # record the rows of the removed scores, 32266 is the number of customers
	p_level = [] # resord the columns of the removed scores
	real = [] # actual score lists
	esti = [] # estimated score lists
	for i in tqdm(c_level):
		for j in range(23812): # 23812 is the number of product
			if df_rs_sys.iloc[i,j]!=0:
				p_level.append(j)
				real.append(df_rs_sys.iloc[i,j])
				df_rs_sys.iloc[i,j] = 0
				c = i
				L = [j]
				esti.append(svdRec.recommend(mat(df_rs_sys.values), c, L,len(L))[0][1])
				df_rs_sys.iloc[i,j] = real[-1]
				break

	abs_err = np.abs(esti - np.array(real)).mean() # absolute error of the estimated recommendation scores
	avg_score = df_rs['score'].mean()

	print('The average actual score of the recommendation system is {:.3f}.\n'.format(avg_score))
	print('The aboluate error of the recommendation system is {:.3f}.\n'.format(abs_error))

