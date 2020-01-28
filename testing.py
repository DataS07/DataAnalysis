#-------------------------------------------------------------------
# @author anilnayak
# Created : 19 / August / 2018 : 9:41 PM
# @copyright (C) 2018, SB.AI, All rights reserved.
#-------------------------------------------------------------------

import random_forest as rf
import load_data as ld
import keras_model as km

# Data Handler object to load the data for training or loading pickle files for testing
ld_obj = ld.DataHandler(pca_decompose=True)

#loading pickle file to get the normalization details mean and std / category information to form the one hot encoding
ld_obj.load_pickle_data()

# Preparing data for testing
data = ld_obj.prepare_live_test_data(category='academic',
									 main_category='comics',
									 currency='usd',
									 deadline='2017-10-10',
									 launched='2016-08-08 12:12:28',
									 goal=0,
									 pledged=0,
									 backers=0,
									 usd_pledged=0,
									 usd_pledged_real=0,
									 usd_goal_real=0)


class Testing():
	def __init__(self, ld_obj):
		self.ld_obj = ld_obj
		self.keras_model = None
		self.random_forest = None
	
	def load_random_forest(self):
		self.random_forest = rf.RandomForest(self.ld_obj)
		self.random_forest.load_model()
	
	def load_nn(self):
		self.keras_model = km.KerasModel(self.ld_obj)
		self.keras_model.load_model_keras()

	def predict(self, x):
		print("Keras Model: ", self.keras_model.predict(x))
		print("Random Forest Model: ", self.random_forest.predict(x))
		
# Small testing app to test all the model
tst = Testing(ld_obj)
# Loading random forest model into application
tst.load_random_forest()
# Loading neural network model into application
tst.load_nn()
# predicting the data prepared for using both the models
tst.predict(data)

