"""Find the association rules between subclass and products.
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
#from numba import jit
from next_basket import read_file, add_transaction_id

from datetime import date as dt
from datetime import datetime

import apriori

dataFolder = 'Data\\'

def item_in_one_basket(df: pd.DataFrame, item_tp: str) -> list:
	"""Group the items in the same basket as a list.
	
	Args:
        df: product purchasing information.
        item_tp: "subclass" or "product_id"
        
    Returns:
        basket_list: a list of items in baskets. Each sub list represents the items in one basket.
	
	"""

	df['trans_id'] =  df['customer_id'].astype(str) + df['date_time'].astype(str) # assume each customer only make at most one transaction everyday
	df = df.sort_values(['trans_id'])
	df['subclass'] = df['subclass'].astype(str)
	row_num = 0
	trans_id_buf = df['trans_id'].values[0]
	basket_list = []
	item_list = []
	for j in tqdm(range(df.shape[0])):
		if df['trans_id'].values[j] == trans_id_buf:
			item_list.append(df['subclass'].values[j])
	else:
		item_list = list(set(item_list))
		basket_list.append(item_list[:])
		item_list = [df['subclass'].values[j]]
		trans_id_buf = df['trans_id'].values[j]

	return basket_list


if __name__ == "__main__":

	print('Reading files ...\n')
	df = read_file(dataFolder)

	# identify the association rule between various subclass
	basket_list = item_in_one_basket(df, 'subclass')
	L, suppData = apriori.apriori(basket_list, minSupport=0.01) # support score is 0.01 --> the persesntage of appearance > 1%
	rules = apriori.generateRules(L, suppData, minConf=0.3) # conditional probability >= 0.3
	print('Association rules in Subclass: \n',rules)

	# identify the association rule between various products
	basket_list = item_in_one_basket(df, 'product_id')
	L, suppData = apriori.apriori(basket_list, minSupport=0.005) # support score is 0.01 --> the persesntage of appearance > 0.5%
	rules = apriori.generateRules(L,suppData,minConf=0.3) # conditional probability >= 0.3
	print('Association rules in Products: \n',L)




