"""Compute the word2vec representation for products.
"""



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
#from numba import jit
from next_basket import read_file, add_transaction_id
from association_rules import item_in_one_basket

import gensim

from datetime import date as dt
from datetime import datetime

from sklearn.decomposition import PCA

dataFolder = 'Data\\'

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """From Tensorflow's tutorial.
    """
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
    	x, y = low_dim_embs[i,:]
    	plt.scatter(x, y)
    	plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
    		ha='right', va='bottom')

    plt.show()



if __name__ == "__main__":

	print('Reading files ...\n')
	df = read_file(dataFolder)

	df['trans_id'] = df['customer_id'].astype(str) + df['date_time'].astype(str)
	baskets = df.groupby("trans_id").apply(lambda order: order['product_id'].astype(str).tolist())
	longest = np.max(baskets.apply(len))
	baskets = baskets.values

	# I choose the window size as the largest basket size.
	# Since there is no sequence characteristics of the products in an order, we should have a 
	# training window huge enough to accommodate all the products together.
	model = gensim.models.Word2Vec(baskets, size=100, window=longest, min_count=2, workers=4)

	# product_id
	vocab = list(model.wv.vocab.keys())

	# reduce dimension using PCA with 2 components
	pca = PCA(n_components=2)
	pca.fit(model.wv.syn0)

	# plot 2d word2vec
	plot_with_labels(pca.fit_transform(model.wv.syn0), vocab)

