OVERVIEW
------------------------------------------------------------------
next_basket.py, association_rules.py, basic_recsys.py, word2vec.py
are the polished pieces of code for Ta-Feng project. Each file can 
run as the main function. 

next_basket.py
------------------------------------------------------------------
Read files from data folder, feature engineering, model training 
for "the next basket prediction" problem.

association_rules.py
------------------------------------------------------------------
Identify the association rules between subclass and products

basic_recsys.py
------------------------------------------------------------------
Build a recommendation system using collaborative filtering. 
Performance is evaluated.

word2vec.py
------------------------------------------------------------------
Create word2vec represesntations of product ids.
In order to reduce the no sequence effect, window size is set to the 
largest basket size in the training set. 
