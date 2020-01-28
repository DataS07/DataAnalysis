#-------------------------------------------------------------------
# @author anilnayak
# Created : 20 / August / 2018 : 1:42 AM
# @copyright (C) 2018, SB.AI, All rights reserved.
#-------------------------------------------------------------------

from sklearn.decomposition import PCA
import pickle
import numpy as np

class PCADecompose():
	def __init__(self, num_feature_to_decompose=50):
		self.pca = PCA(n_components=num_feature_to_decompose)
		
	def transform_data(self, data, data_y):
		self.pca.fit(data)
		self.pca.get_precision()
		print(self.pca.explained_variance_ratio_)
		data1_new = self.pca.transform(data)
		data_new = np.hstack((data1_new, data_y))
		
		pickle.dump(self.pca, open("pca.pkl", "wb"))
		
		return data_new
	
	def transform(self,data):
		return self.pca.transform(data)
	
