#-------------------------------------------------------------------
# @author anilnayak
# Created : 19 / August / 2018 : 7:38 PM
# @copyright (C) 2018, SB.AI, All rights reserved.
#-------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import os.path
import numpy as np
class RandomForest():
	def __init__(self, data, model_name='random_forest_model_features.pkl'):
		self.data = data
		self.model_name = model_name
		self.model_loaded = False
		self.clf_RandomForest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2,
													   random_state=0)
	def train(self):
		self.clf_RandomForest.fit(self.data.X_train, np.squeeze(self.data.Y_train))
		
	def features_val(self):
		print(self.clf_RandomForest.feature_importances_[0:10])
		
	def validate(self):
		scores = cross_val_score(self.clf_RandomForest, self.data.X_validation, self.data.Y_validation)
		print(scores)
		
	def test(self):
		scores_test = cross_val_score(self.clf_RandomForest, self.data.X_test, self.data.Y_test)
		print(scores_test)
	
	def save(self):
		joblib.dump(self.clf_RandomForest, self.model_name)
		print('model saved successfully')
		
	def load_model(self):
		if os.path.isfile(self.model_name):
			self.clf_RandomForest = joblib.load(self.model_name)
			self.model_loaded = True
			print('model restored successfully')
		else:
			print("Model file does not exist !!!")
	
	def predict(self, x_p):
		if self.model_loaded:
			pred = self.clf_RandomForest.predict(x_p)
			return int(pred[0])
		else:
			print("Model does not exist !!!")

	