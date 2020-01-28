#-------------------------------------------------------------------
# @author anilnayak
# Created : 19 / August / 2018 : 1:39 PM
# @copyright (C) 2018, SB.AI, All rights reserved.
#-------------------------------------------------------------------
import numpy as np
import tensorflow as tf

class TensorflowModel():
	def __init__(self, data, features, classes, hidden_layers_units, batch_size=100, epochs=5, learning_rate=0.001 ):
		self.features = features
		self.classes = classes
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.hidden_layers_units = hidden_layers_units
		self.batch_size = batch_size
		self.cost_history = np.empty(shape=[1], dtype=float)
		self.is_training = tf.Variable(True, dtype=tf.bool)
		self.data = data
	
	def prepare_model(self):
		self.X = tf.placeholder(tf.float32, [None, self.features],name='input_tensor')
		self.Y = tf.placeholder(tf.float32, [None, self.classes], name='output_tensor')
		
		initializer = tf.contrib.layers.xavier_initializer()
		layers = [self.X]
		
		for hidden in range(len(self.hidden_layers_units)):
			hidden_units = self.hidden_layers_units[hidden]
			layers_hidden = tf.layers.dense(layers[-1], hidden_units, activation=tf.nn.relu, kernel_initializer=initializer,
											name="hidden_"+str(hidden))
			layers.append(layers_hidden)
			
		self.output_layer = tf.layers.dense(layers[-1], self.classes, activation=None)
		self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.output_layer)
		self.cost = tf.reduce_mean(self.cross_entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		
		self.predicted = tf.nn.sigmoid(self.output_layer)
		self.correct_pred = tf.equal(tf.round(self.predicted), self.Y)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
	
	def training_model(self):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.sess = sess
			for epoch in range(self.epochs):
				epoch_loss = 0
				start = 0
				
				for batch in range(len(self.data.X_train) // self.batch_size):
					batch_x = self.data.X_train[start:start + self.batch_size]
					batch_y = self.data.Y_train[start:start + self.batch_size]
					start += self.batch_size
					c, _, acc = sess.run([self.cost, self.optimizer, self.accuracy], feed_dict={self.X: batch_x, self.Y: batch_y})
					epoch_loss += c
				
				self.cost_history = np.append(self.cost_history, epoch_loss)
				acc, p = sess.run([self.accuracy, tf.round(self.predicted)], feed_dict={self.X: self.data.X_validation, self.Y: self.data.Y_validation})
				print('Epoch: ', epoch, 'Loss: ', epoch_loss, 'Validation Accuracy:', acc)
	
	def test_model(self):
		try:
			acc, p = self.sess.run([self.accuracy, tf.round(self.predicted)], feed_dict={self.X: self.data.X_test,self.Y: self.data.Y_test})
			print('Testing Accuracy:', acc)
		except:
			''
		
	