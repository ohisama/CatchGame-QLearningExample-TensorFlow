import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import tensorflow as tf
import numpy as np
import math
import time
import math
import os
import random

class CatchEnvironment():
	def __init__(self, gridSize):
		self.gridSize = gridSize
		self.nbStates = self.gridSize * self.gridSize
		self.state = np.empty(3, dtype = np.uint8)
	def observe(self):
		canvas = self.drawState()
		canvas = np.reshape(canvas, (-1,self.nbStates))
		return canvas
	def drawState(self):
		canvas = np.zeros((self.gridSize, self.gridSize))
		canvas[self.state[0] - 1, self.state[1] - 1] = 1
		canvas[self.gridSize - 1, self.state[2] -1 - 1] = 1
		canvas[self.gridSize - 1, self.state[2] -1] = 1
		canvas[self.gridSize - 1, self.state[2] -1 + 1] = 1
		return canvas
	def reset(self):
		initialFruitColumn = random.randrange(1, self.gridSize + 1)
		initialBucketPosition = random.randrange(2, self.gridSize + 1 - 1)
		self.state = np.array([1, initialFruitColumn, initialBucketPosition])
		return self.getState()
	def getState(self):
		stateInfo = self.state
		fruit_row = stateInfo[0]
		fruit_col = stateInfo[1]
		basket = stateInfo[2]
		return fruit_row, fruit_col, basket
	def getReward(self):
		fruitRow, fruitColumn, basket = self.getState()
		if (fruitRow == self.gridSize - 1):
			if (abs(fruitColumn - basket) <= 1):
				return 1
			else:
				return -1
		else:
			return 0
	def isGameOver(self):
		if (self.state[0] == self.gridSize - 1):
			return True
		else:
			return False
	def updateState(self, action):
		if (action == 1):
			action = -1
		elif (action == 2):
			action = 0
		else:
			action = 1
		fruitRow, fruitColumn, basket = self.getState()
		newBasket = min(max(2, basket + action), self.gridSize - 1)
		fruitRow = fruitRow + 1
		self.state = np.array([fruitRow, fruitColumn, newBasket])
	def act(self, action):
		self.updateState(action)
		reward = self.getReward()
		gameOver = self.isGameOver()
		return self.observe(), reward, gameOver, self.getState()

class Test(QWidget):
	def __init__(self):
		app = QApplication(sys.argv)
		super().__init__()
		self.init_ui()
		self.show()
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.update)
		self.timer.start(200)
		self.winCount = 0
		self.loseCount = 0
		app.exec_()
	def init_ui(self):
		self.setWindowTitle("PyQt5")
		self.resize(400, 400)
		self.angle = 0
	def paintEvent(self, QPaintEvent):
		currentState = env.observe()
		q = sess.run(output_layer, feed_dict = {
			X: currentState
		})
		index = q.argmax()
		action = index + 1
		nextState, reward, gameOver, stateInfo = env.act(action)
		fruitRow = stateInfo[0]
		fruitColumn = stateInfo[1]
		basket = stateInfo[2]
		if (reward == 1):
			self.winCount = self.winCount + 1
		elif (reward == -1):
			self.loseCount = self.loseCount + 1
		painter = QPainter(self)
		painter.setPen(Qt.black)
		painter.drawLine(QPoint(20, 20), QPoint(20, 220))
		painter.drawLine(QPoint(20, 20), QPoint(220, 20))
		painter.drawLine(QPoint(220, 20), QPoint(220, 220))
		painter.drawLine(QPoint(20, 220), QPoint(220, 220))
		painter.setFont(QFont('Consolas', 20))
		painter.drawText(QPoint(250, 50), "win: " + str(self.winCount));
		painter.drawText(QPoint(250, 80), "miss: " + str(self.loseCount));
		painter.setBrush(Qt.yellow)
		painter.drawRect(fruitColumn * 20, fruitRow * 20, 20, 20)
		painter.setBrush(Qt.green)
		painter.drawRect(basket * 20, 10 * 20, 20, 20)
		if (gameOver):
			fruitRow, fruitColumn, basket = env.reset()

nbActions = 3
hiddenSize = 100
batchSize = 50
gridSize = 10
nbStates = gridSize * gridSize
learningRate = 0.2
X = tf.placeholder(tf.float32, [None, nbStates])
Y = tf.placeholder(tf.float32, [None, nbActions])
W1 = tf.Variable(tf.truncated_normal([nbStates, hiddenSize], stddev = 1.0 / math.sqrt(float(nbStates))))
b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev = 0.01))
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize], stddev = 1.0 / math.sqrt(float(hiddenSize))))
b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev = 0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)
W3 = tf.Variable(tf.truncated_normal([hiddenSize, nbActions], stddev = 1.0 / math.sqrt(float(hiddenSize))))
b3 = tf.Variable(tf.truncated_normal([nbActions], stddev = 0.01))
output_layer = tf.matmul(hidden_layer, W3) + b3
cost = tf.reduce_sum(tf.square(Y - output_layer)) / (2 * batchSize)
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
env = CatchEnvironment(gridSize)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, os.getcwd() + "/model.ckpt")
fruitRow, fruitColumn, basket = env.reset()

if __name__ == '__main__':
	Test()



