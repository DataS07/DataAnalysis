# -------------------------------------------------------------------
# @author anilnayak
# Created : 17 / August / 2018 : 10:55 AM
# @copyright (C) 2018, SB.AI, All rights reserved.
# -------------------------------------------------------------------
import numpy as np
import load_data as ld
import tensorflow_model as tfm
import keras_model as km
import random_forest as rf

class Train():
	def __init__(self, file_name, filter_attribute, pca_decompose=False, num_feature_to_decompose=50):
		self.ld_obj = ld.DataHandler(pca_decompose=pca_decompose,num_feature_to_decompose=num_feature_to_decompose)
		self.file_name = file_name
		self.filter_attribute = filter_attribute
		self.features = None
		self.classes = None
	
	def loaddata(self):
		# Load Data
		self.ld_obj.load_data(self.file_name, self.filter_attribute)
	
	def normalize_continuous_data(self):
		# Normalize Continuous Data
		self.ld_obj.normalize_continuous_data()
	
	def one_hot_encoding_categorical_data(self, categories, date_attributes, target='state',
										  success_class='successful'):
		# one hot encoding categorical data
		self.ld_obj.one_hot_encoding_categorical_data(categories, date_attributes)
		self.ld_obj.one_hot_encode_target_value(target=target,success_class=success_class)
	
	def spilt_data(self, test_data_perc=0.2):
		# split data
		self.ld_obj.prepare_data(test_data_perc)
		self.features = np.shape(self.ld_obj.X_train)[1]
		self.classes = np.shape(self.ld_obj.Y_train)[1]
		
	def tensorflow_model(self):
		tfmobj = tfm.TensorflowModel(self.ld_obj, self.features, self.classes, [self.features-1])
		tfmobj.prepare_model()
		tfmobj.training_model()
		tfmobj.test_model()
		
	def keras_model(self):
		kmobj = km.KerasModel(self.ld_obj, self.features, self.classes, [self.features-10,self.features//2])
		kmobj.prepare_model()
		kmobj.training_model()
		kmobj.test_model()
		kmobj.save_model()
		kmobj.show_loss_curve()
		kmobj.show_accurcy_curve()
		
	def random_forest(self):
		rfobj = rf.RandomForest(self.ld_obj)
		rfobj.train()
		rfobj.validate()
		rfobj.test()
		rfobj.save()
		rfobj.features_val()


file_name = 'data.csv'
filter_attribute = ['category', 'main_category', 'currency', 'deadline', 'goal', 'launched', 'pledged', 'state',
					'backers', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']

training_obj = Train(file_name, filter_attribute,pca_decompose=True, num_feature_to_decompose=100)
training_obj.loaddata()
training_obj.normalize_continuous_data()
categories = ['category', 'main_category', 'currency']
date_attributes = ['deadline', 'launched']
training_obj.one_hot_encoding_categorical_data(categories, date_attributes, target='state', success_class='successful')
print(training_obj.ld_obj.data.head())
training_obj.spilt_data()

print("Training Random Forest model")
training_obj.random_forest()

print("Training Tensorflow model ")
training_obj.tensorflow_model()

print("Training Keras model ")
training_obj.keras_model()




# clf_RandomForest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
# clf_RandomForest.fit(X_train, Y_train)
# clf_RandomForest.feature_importances_[0:10]
# scores = cross_val_score(clf_RandomForest, X_validation, Y_validation)
# scores_test = cross_val_score(clf_RandomForest, X_test, Y_test)
# joblib.dump(clf_RandomForest, 'random_forest_model_385_features.pkl')

# import MLP
# nn = MLP.MLP()
# nn.neural_network_model(nn.x)
# nn.train(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)


# # print('Startng Training rbf')
# # clf = svm.SVC(C=10, kernel='rbf',
# # 			  gamma=0.001,verbose=False,
# # 			  decision_function_shape='ovo',
# # 			  max_iter=100)
# # clf.fit(X_train, Y_train)
# # print("RBF Score",clf.score(X_test, Y_test))
#
# print('Startng Training Linear')
# lin_clf = svm.LinearSVC(C=100, verbose=True, max_iter=10)
# lin_clf.fit(X_train, Y_train)
# print("Linear Score", lin_clf.score(X_test, Y_test))

# # print('Starting Gaussian Model')
# # from sklearn.naive_bayes import GaussianNB
# # gnb = GaussianNB()
# # gnb.fit(X_train, Y_train)
# # print("gnb Score",gnb.score(X_test, Y_test))
#
# print('Starting decision tree')
# from sklearn import tree
#
# tree_clf = tree.DecisionTreeClassifier()
# tree_clf = tree_clf.fit(X_train, Y_train)
# print("decision tree Score", tree_clf.score(X_test, Y_test))
#

# print("Starting adaboost")
# from sklearn.ensemble import AdaBoostClassifier
# clf_ada = AdaBoostClassifier(n_estimators=100)
# scores_ada = cross_val_score(clf_ada, X_train, Y_train)
# print("adaboost score ",scores_ada)


# from sklearn.svm import SVC
# from sklearn.multiclass import OneVsRestClassifier
# clf_one = OneVsRestClassifier(estimator=SVC(C=100,random_state=0,max_iter=100))
# clf_one.fit(X_train, Y_train)
# print("One vs rest ",clf_one.score(X_test, Y_test))


# from sklearn.externals import joblib
# joblib.dump(clf, 'filename.pkl')

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# accuracy_score(y_true, y_pred)

# from sklearn.metrics import classification_report
# y_true = [0, 1, 2, 2, 0]
# y_pred = [0, 0, 2, 1, 0]
# target_names = ['class 0', 'class 1', 'class 2']
# print(classification_report(y_true, y_pred, target_names=target_names))
#              precision    recall  f1-score   support

# from sklearn.metrics import jaccard_similarity_score
# jaccard_similarity_score(y_true, y_pred)
# jaccard_similarity_score(y_true, y_pred, normalize=False)
