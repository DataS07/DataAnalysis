# -------------------------------------------------------------------
# @author anilnayak
# Created : 19 / August / 2018 : 6:03 PM
# @copyright (C) 2018, SB.AI, All rights reserved.
# -------------------------------------------------------------------
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import collections
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import PCA

class DataHandler():
	def __init__(self, pca_decompose=False, num_feature_to_decompose=50):
		self.normalizatino_pkl_file = "mean_std.pkl"
		self.categorical_pkl_file = "categorical.pkl"
		self.data = None
		self.continuous_data = None
		self.categorical_data = None
		self.continuous_data_before_norm = None
		self.dict_categories = {}
		self.category_data_pkl = None
		self.mean_std_data_pkl = None
		self.pca_decompose = pca_decompose
		self.num_feature_to_decompose = num_feature_to_decompose
		self.PCAObj = None
	
	def load_data(self, file_name, keep_attribute):
		"""
		Load and Filter data into categorical and continuous dataframe
		:param file_name:
		:param filter_attribute:
		:return:
		"""
		data_frame = pd.read_csv(file_name)
		data_frame = data_frame[keep_attribute]
		
		data_frame = data_frame.fillna(0.0)
		data_frame.isnull().any()
		self.categorical_data = data_frame.select_dtypes(include=['object']).copy()
		self.continuous_data = data_frame.select_dtypes(include=['float', 'int']).copy()
		
		self.continuous_data_before_norm = self.continuous_data.copy()
	
	def normalize_continuous_data(self):
		mean_data = np.mean(self.continuous_data).values
		std_data = np.std(self.continuous_data).values
		self.continuous_data = (self.continuous_data.values - mean_data) / std_data
		self.continuous_data = pd.DataFrame({'goal': self.continuous_data[:, 0],
											'pledged': self.continuous_data[:, 1],
											'backers': self.continuous_data[:, 2],
											'usd pledged': self.continuous_data[:, 3],
											'usd_pledged_real': self.continuous_data[:, 4],
											'usd_goal_real': self.continuous_data[:, 5]})
		self.continuous_data = self.continuous_data[['goal', 'pledged', 'backers', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']]
		file_data = {"mean_data": mean_data, "std_data": std_data}
		pickle.dump(file_data, open(self.normalizatino_pkl_file, "wb"))
		self.mean_std_data_pkl = file_data
		if self.data is None:
			self.data = self.continuous_data
	
	def scatter_plot(self):
		scatter_matrix(self.continuous_data[['goal', 'pledged', 'backers', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']])
	
	def one_hot_encoding_categorical_data(self, categories, date_attributes):
		if self.data is None:
			print('Prepare continuous data prior to categorical data')
			return
		
		for cat in categories:
			self.one_hot_encoding(None, cat)
		
		for attr in date_attributes:
			data_frame = pd.DataFrame(self.categorical_data[attr].str.split('-').tolist(), columns=['year_' + attr,
																									'month_' + attr,
																									'date'])[
				['year_'+attr, 'month_'+attr]]

			self.one_hot_encoding(data_frame, 'year_' + attr)
			self.one_hot_encoding(data_frame, 'month_' + attr)
		
		pickle.dump(self.dict_categories, open(self.categorical_pkl_file, "wb"))
		
		self.category_data_pkl = self.dict_categories
		
	
	def one_hot_encode_target_value(self, target='state', success_class='successful'):
		self.data = pd.concat([self.data, self.categorical_data[target]], axis=1)
		values = self.categorical_data[target].astype('category').cat.categories.tolist()
		state_dict = collections.defaultdict(int)
		state_map_comp = {}
		for i in values:
			if i.lower() == success_class:
				j = 1
			else:
				j = 0
			state_dict[i.lower()] = j
		self.dict_categories[target] = dict(state_dict)
		state_map_comp[target] = dict(state_dict)
		self.data.replace(state_map_comp, inplace=True)
	
	def one_hot_encoding(self, data_frame, name):
		if data_frame is not None:
			cat_data = data_frame[name]
		else:
			cat_data = self.categorical_data[name]
		lb = LabelBinarizer()
		lb_results = lb.fit_transform(cat_data)
		lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)
		
		self.data = pd.concat([self.data, lb_results_df], axis=1)
		
		classes = lb.classes_
		dict_data = collections.defaultdict(int)
		j = 0
		for i in classes:
			dict_data[i.lower()] = j
			j += 1
		self.dict_categories[name] = dict(dict_data)
		
	def prepare_data(self, test_data_perc=0.2):
		self.data = shuffle(self.data)
		self.data_np_arr1 = self.data.values
		self.features = np.shape(self.data_np_arr1)[1] - 1
		
		if self.pca_decompose:
			self.PCAObj = PCA.PCADecompose(self.num_feature_to_decompose)
			d_x = self.data_np_arr1[:, 0:self.features]
			d_y = self.data_np_arr1[:, self.features:]
			data_new = self.PCAObj.transform_data(d_x,d_y)
			self.data_np_arr = data_new
			self.features = np.shape(self.data_np_arr)[1] - 1
		else:
			self.data_np_arr = self.data_np_arr1

		train, test = train_test_split(self.data_np_arr, test_size=test_data_perc)
		self.X_train = train[:, 0:self.features]
		self.Y_train = train[:, self.features:]
		validation, test = train_test_split(test, test_size=0.5)
		self.X_validation = validation[:, 0:self.features]
		self.Y_validation = validation[:, self.features:]
		self.X_test = test[:, 0:self.features]
		self.Y_test = test[:, self.features:]
		
		print("X Train shape ", np.shape(self.X_train))
		print("Y Train shape ", np.shape(self.Y_train))
		print("X Validation shape ", np.shape(self.X_validation))
		print("Y Validation shape ", np.shape(self.Y_validation))
		print("X Test shape ", np.shape(self.X_test))
		print("Y Test shape ", np.shape(self.Y_test))
	
	def correlation(self):
		print(self.continuous_data_before_norm.corr())
	
	def load_pickle_data(self):
		file = open('categorical.pkl', 'rb')
		self.category_data_pkl = pickle.load(file)
		file1 = open('mean_std.pkl', 'rb')
		self.mean_std_data_pkl = pickle.load(file1)
		file2 = open("pca.pkl", 'rb')
		self.PCAObj = pickle.load(file2)
	
	def prepare_live_test_data(self, category, main_category, currency, deadline,
				launched, goal=0, pledged=0, backers=0, usd_pledged=0, usd_pledged_real=0, usd_goal_real=0):
		numerics_data = ([goal, pledged, backers, usd_pledged, usd_pledged_real, usd_goal_real] -
						 self.mean_std_data_pkl['mean_data']) / self.mean_std_data_pkl['std_data']
		
		category_val = [0] * len(list(self.category_data_pkl['category']))
		category_val[int(self.category_data_pkl['category'][category])] = 1
		
		main_category_val = [0] * len(list(self.category_data_pkl['main_category']))
		main_category_val[int(self.category_data_pkl['main_category'][main_category])] = 1
		
		currency_val = [0] * len(list(self.category_data_pkl['currency']))
		currency_val[int(self.category_data_pkl['currency'][currency])] = 1
		
		d_arr = deadline.split("-")
		year_deadline_val = [0] * len(list(self.category_data_pkl['year_deadline']))
		year_deadline_val[int(self.category_data_pkl['year_deadline'][d_arr[0]])] = 1
		
		month_deadline_val = [0] * len(list(self.category_data_pkl['month_deadline']))
		month_deadline_val[int(self.category_data_pkl['month_deadline'][d_arr[1]])] = 1
		
		l_arr = launched.split("-")
		year_launched_val = [0] * len(list(self.category_data_pkl['year_launched']))
		year_launched_val[int(self.category_data_pkl['year_launched'][l_arr[0]])] = 1
		
		month_launched_val = [0] * len(list(self.category_data_pkl['month_launched']))
		month_launched_val[int(self.category_data_pkl['month_launched'][l_arr[1]])] = 1
		
		data_o = list(numerics_data) + list(category_val) + list(main_category_val) + list(currency_val) + list(
			year_deadline_val) + \
				 list(month_deadline_val) + list(year_launched_val) + list(month_launched_val)
		data_o = np.array([np.array(data_o, dtype='float')])
		
		if self.pca_decompose:
			data_o = np.squeeze(self.PCAObj.transform(data_o)[0])
			data_o = np.array([np.array(data_o, dtype='float')])
		
		return data_o