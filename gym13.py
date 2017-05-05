from __future__ import print_function
import math
import sys
import gym
import gym.spaces
import numpy as np
from gym import core, spaces
from gym.utils import seeding
from numpy import sin, cos, pi
import time
import random
import tensorflow as tf
import os

class FBEnvironment(core.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 30
	}
	def __init__(self):
		self.viewer = None
		self.gridSize = 10
		self.nbStates = self.gridSize * self.gridSize
		self.state = np.empty(3, dtype = np.uint8)

	def drawState(self):
		canvas = np.zeros((self.gridSize, self.gridSize))
		canvas[self.state[0] - 1, self.state[1] - 1] = 1
		canvas[self.gridSize - 1, self.state[2] - 1 - 1] = 1
		canvas[self.gridSize - 1, self.state[2] - 1] = 1
		canvas[self.gridSize - 1, self.state[2] - 1 + 1] = 1
		return canvas
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
	def observe(self):
		canvas = self.drawState()
		canvas = np.reshape(canvas, (-1, self.nbStates))
		return canvas
	def _reset(self):
		initialFruitColumn = random.randrange(1, self.gridSize + 1)
		initialBucketPosition = random.randrange(2, self.gridSize + 1 - 1)
		self.state = np.array([1, initialFruitColumn, initialBucketPosition])
		return self.observe()
	def _step(self, action):
		self.updateState(action)
		reward = self.getReward()
		gameOver = self.isGameOver()
		return self.observe(), reward, gameOver, {}
	def _render(self, mode = 'human', close = False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return
		from gym.envs.classic_control import rendering
		if self.viewer is None:
			self.viewer = rendering.Viewer(500, 500)
			self.viewer.set_bounds(-2.5, 2.5, -2.5, 2.5)
		x = -1.8 + (self.state[1] - 1) * 0.4
		y = 2.0 - (self.state[0] - 1) * 0.4
		z = -1.8 + (self.state[2] - 1) * 0.4
		transform0 = rendering.Transform(translation = (x, y))
		transform1 = rendering.Transform(translation = (z, -1.8))
		self.viewer.draw_circle(0.2, 20, color = (1, 1, 0)).add_attr(transform0)
		self.viewer.draw_line((-2.0, 2.0), (2.0, 2.0), color = (0, 0, 0))
		self.viewer.draw_line((-2.0, 2.0), (-2.0, -2.0), color = (0, 0, 0))
		self.viewer.draw_line((2.0, 2.0), (2.0, -2.0), color = (0, 0, 0))
		self.viewer.draw_line((-2.0, -2.0), (2.0, -2.0), color = (0, 0, 0))
		self.viewer.draw_polygon([(-0.6, -0.2), (0.6, -0.2), (0.6, 0.2), (-0.6, 0.2)], color = (0, 1, 0)).add_attr(transform1)
		return self.viewer.render(return_rgb_array = mode == 'rgb_array')

gridSize = 10
nbStates = gridSize * gridSize
nbActions = 3
hiddenSize = 100
batchSize = 50
learningRate = 0.2
maxMemory = 500
discount = 0.9
epoch = 1001
epsilon = 1
epsilonMinimumValue = 0.001
X = tf.placeholder(tf.float32, [None, nbStates])
Y = tf.placeholder(tf.float32, [None, nbActions])
W1 = tf.Variable(tf.truncated_normal([nbStates, hiddenSize], stddev = 1.0 / math.sqrt(float(nbStates))))
b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev = 0.01))
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize],stddev = 1.0 / math.sqrt(float(hiddenSize))))
b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev = 0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)
W3 = tf.Variable(tf.truncated_normal([hiddenSize, nbActions],stddev = 1.0 / math.sqrt(float(hiddenSize))))
b3 = tf.Variable(tf.truncated_normal([nbActions], stddev = 0.01))
output_layer = tf.matmul(hidden_layer, W3) + b3
cost = tf.reduce_sum(tf.square(Y - output_layer)) / (2 * batchSize)
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

def randf(s, e):
	return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;

class ReplayMemory:
	def __init__(self, gridSize, maxMemory, discount):
		self.maxMemory = maxMemory
		self.gridSize = gridSize
		self.nbStates = self.gridSize * self.gridSize
		self.discount = discount
		self.inputState = np.empty((self.maxMemory, 100), dtype = np.float32)
		self.actions = np.zeros(self.maxMemory, dtype = np.uint8)
		self.nextState = np.empty((self.maxMemory, 100), dtype = np.float32)
		self.gameOver = np.empty(self.maxMemory, dtype = np.bool)
		self.rewards = np.empty(self.maxMemory, dtype = np.int8)
		self.count = 0
		self.current = 0
	def remember(self, currentState, action, reward, nextState, gameOver):
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.inputState[self.current, ...] = currentState
		self.nextState[self.current, ...] = nextState
		self.gameOver[self.current] = gameOver
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.maxMemory
	def getBatch(self, model, batchSize, nbActions, nbStates, sess, X):
		memoryLength = self.count
		chosenBatchSize = min(batchSize, memoryLength)
		inputs = np.zeros((chosenBatchSize, nbStates))
		targets = np.zeros((chosenBatchSize, nbActions))
		for i in range(chosenBatchSize):
			if memoryLength == 1:
				memoryLength = 2
			randomIndex = random.randrange(1, memoryLength)
			current_inputState = np.reshape(self.inputState[randomIndex], (1, 100))
			target = sess.run(model, feed_dict = {
				X: current_inputState
			})
			current_nextState = np.reshape(self.nextState[randomIndex], (1, 100))
			current_outputs = sess.run(model, feed_dict = {
				X: current_nextState
			})
			nextStateMaxQ = np.amax(current_outputs)
			if (self.gameOver[randomIndex] == True):
				target[0, [self.actions[randomIndex] - 1]] = self.rewards[randomIndex]
			else:
				target[0, [self.actions[randomIndex] - 1]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ
			inputs[i] = current_inputState
			targets[i] = target
		return inputs, targets

def main(_):
	print ("Training new model")
	env = FBEnvironment()
	memory = ReplayMemory(gridSize, maxMemory, discount)
	saver = tf.train.Saver()
	winCount = 0
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		for i in range(epoch):
			err = 0
			currentState = env.reset()
			isGameOver = False
			while (isGameOver != True):
				action = 0
				global epsilon, output_layer, optimizer, cost, batchSize, nbActions, nbStates, X
				if (randf(0, 1) <= epsilon):
					action = random.randrange(0, nbActions - 1)
				else:
					q = sess.run(output_layer, feed_dict = {
						X: currentState
					})
					action = q.argmax()
					action += 1
				if (epsilon > epsilonMinimumValue):
					epsilon = epsilon * 0.999
				nextState, reward, gameOver, _ = env.step(action)
				env.render()
				time.sleep(0.1)
				if (reward == 1):
					winCount += 1
				memory.remember(currentState, action, reward, nextState, gameOver)
				currentState = nextState
				isGameOver = gameOver
				inputs, targets = memory.getBatch(output_layer, batchSize, nbActions, nbStates, sess, X)
				_, loss = sess.run([optimizer, cost], feed_dict = {
					X: inputs,
					Y: targets
				})
				err += loss
			print("Epo: " + str(i) + " loss: " + str(err) + " count: " + str(winCount) + " ratio: " + str(float(winCount) / float(i + 1) * 100))
		save_path = saver.save(sess, os.getcwd() + "/model2.ckpt")
		print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
	tf.app.run()



