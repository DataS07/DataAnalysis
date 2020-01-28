# -------------------------------------------------------------------
# @author anilnayak
# Created : 19 / August / 2018 : 1:39 PM
# @copyright (C) 2018, SB.AI, All rights reserved.
# -------------------------------------------------------------------
import numpy as np
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.models import Model
from keras.models import load_model
import os.path
import matplotlib.pyplot as plt

class KerasModel():
	def __init__(self, data=None, features=None, classes=None, hidden_layers_units=None, batch_size=100, epochs=5, learning_rate=0.001,
				 optimizer='adam', loss='binary_crossentropy',model_name='model_keras.h5'):
		self.features = features
		self.classes = classes
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.hidden_layers_units = hidden_layers_units
		self.batch_size = batch_size
		self.cost_history = np.empty(shape=[1], dtype=float)
		self.data = data
		self.keras_model = None
		self.optimizer = optimizer
		self.loss = loss
		self.model_name = model_name
		self.model_loaded = False
		self.history = None
		if data:
			self.category_data_pkl = data.category_data_pkl
			self.mean_std_data_pkl = data.mean_std_data_pkl
		else:
			self.category_data_pkl = None
			self.mean_std_data_pkl = None
	
	def show_loss_curve(self):
		if self.history:
			v_loss = self.history.history['loss']
			v_val_loss = self.history.history['val_loss']
			epochs = range(1, len(v_loss) + 1)
			plt.plot(epochs, v_loss, 'bo', label='Training loss')
			plt.plot(epochs, v_val_loss, 'b', label='Validation loss')
			plt.title('Training and validation loss')
			plt.xlabel('Epochs')
			plt.ylabel('Loss')
			plt.legend()
			plt.show()
		else:
			print('Training is not done !!!')
			
	def show_accurcy_curve(self):
		if self.history:
			acc = self.history.history['acc']
			val_acc = self.history.history['val_acc']
			epochs = range(1, len(acc) + 1)
			plt.plot(epochs, acc, 'bo', label='Training acc')
			plt.plot(epochs, val_acc, 'b', label='Validation acc')
			plt.title('Training and validation accuracy')
			plt.xlabel('Epochs')
			plt.ylabel('Accuracy')
			plt.legend()
			plt.show()
		else:
			print('Training is not done !!!')
	
	def load_model_keras(self):
		if os.path.isfile(self.model_name):
			self.keras_model = load_model(self.model_name)
			self.model_loaded = True
			print('model restored successfully')
		else:
			print("Model file does not exist !!!")
	
	def save_model(self):
		if self.keras_model is not None:
			self.keras_model.save(self.model_name)
			print('model saved successfully')
		else:
			print("Model not defined !!!")
	
	def model_summary(self):
		if self.keras_model is not None:
			self.keras_model.summary()
		else:
			print("Model not defined !!!")
		
	
	def prepare_model(self):
		self.X = Input(shape=[self.features], name='input_tensor')
		
		layers = [self.X]
		
		for hidden in range(len(self.hidden_layers_units)):
			hidden_units = self.hidden_layers_units[hidden]
			layer = Dense(units=hidden_units, name='fc_'+str(hidden), activation='relu', kernel_initializer='random_uniform')(
				layers[-1])
			layer = BatchNormalization()(layer)
			layers.append(layer)
			
		out_put = Dense(units=self.classes, name='out_put', activation=None)(layers[-1])
		out = Activation('sigmoid')(out_put)
		layers.append(out)
		self.keras_model = Model(inputs=[layers[0]], outputs=[layers[-1]])
		self.keras_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
	
		
	
	def training_model(self):
		self.history = self.keras_model.fit(self.data.X_train,
									   self.data.Y_train,
									   epochs=self.epochs,
									   batch_size=self.batch_size,
									   validation_data=(self.data.X_validation, self.data.Y_validation),
									   verbose=1)
	
	def test_model(self):
		self.testing_accurcy = self.keras_model.evaluate(self.data.X_test, self.data.Y_test)
		print('Testing Accuracy:', self.testing_accurcy)
	
	def predict(self, x_p, threshold=0.8):
		if self.keras_model:
			prediction_p = self.keras_model.predict(x_p, batch_size=None, verbose=0, steps=None)
			return [1 if prediction_p[i][0] >= threshold else 0 for i in range(len(prediction_p))]
		else:
			print("Model does not exist !!!")
	


