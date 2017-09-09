# -*- coding: utf-8 -*-
import numpy as np

class LogisticRegression():
	def __init__(self):
		self.theta_n = []
		self.theta_0 = 0.
		self.loss = []

	def sigmoid(self, x):
		return (1/(1+np.exp(-x)))

	#inicializa os pesos aleatoriamente com amostras da distribuição normal
	def init_weights(self, dim):
		return np.random.randn(dim).reshape(dim,1)
		#return np.ones(dim).reshape(dim,1)

	#função de custo: cross-entropy
	def loss_function(self, Y, sigmoid_z, m):
		loss = -np.sum(np.multiply(Y,np.log(sigmoid_z)) + np.multiply(1-Y,np.log(1-sigmoid_z)))/m

		return loss

	def prints(self, epoch):
		print "--epoca %s: " % epoch
		print "loss: ", self.loss[epoch]
		print "theta: ", self.theta_0.reshape(theta[0].shape[0]), self.theta_n.reshape(theta[1].shape[0])


	def fit(self, X, Y, epochs=3, learning_rate=0.01, print_results=False):
		#dimensão dos dados
		m = X.shape[0]
		n = X.shape[1]

		#inicializa os pesos aleatoriamente
		self.theta_n = self.init_weights(n)
		self.theta_0 = self.init_weights(1)
		X_orig = X
		X = X.T
		Y = Y.reshape(1,m)

		#verifica as dimensões
		#assert(self.theta_n.shape[0] == X.shape[0])
		
		for i in xrange(epochs):
			#calcula Z
			Z = np.dot(self.theta_n.T, X) + self.theta_0

			#calcula gradientes
			sigmoid_z = self.sigmoid(Z)	#função de ativação

			gZ = sigmoid_z - Y
			
			gTheta_n = np.dot(X, gZ.T)/m
			gTheta_0 = np.sum(gZ)/m

			#calcula função de custo
			loss = self.loss_function(Y, sigmoid_z, m)
			self.loss.append(loss)

			#atualiza pesos
			self.theta_0 -= learning_rate*gTheta_0
			self.theta_n -= learning_rate*gTheta_n

			if print_results:
				self.prints(i)

		return self

	def accuracy_score(self, X, Y):
		m = X.shape[0]
		Y_pred = self.predict(X)
		#número de exemplos menos o número de erros dividido pelo número de exemplos
		accuracy =  float(m - np.sum(np.logical_xor(Y_pred, Y)))/m

		return accuracy


	def predict(self, X):
		X = X.T

		#verifica as dimensões antes de fazer o produto interno
		#assert(self.theta_n.shape[0] == X.shape[0])

		Z = np.dot(self.theta_n.T, X) + self.theta_0
		sigmoid_z = self.sigmoid(Z)	#função de ativação

		#Z.shape == (1,m)
		#sigmoid_z.shape = (1,m) -> todas as predições estão neste array

		#verifica se cada predição é maior ou igual a 0.5 e atribui classe 0 ou 1
		Y_predict = np.greater_equal(sigmoid_z, 0.5)

		return Y_predict.astype(int).flatten()


