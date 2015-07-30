import numpy as np

class HMM:
	def __init__(self, states, observables):
		# Set states and observables
		self.numStates = states
		self.numObservables = observables
		self.randomizeProbabilities()

	def randomizeProbabilities(self):
		# Alpha are state probabilities
		# Also called the "transition probabilities"
		# Alpha[i][j]: probability of state j at t+1 given i at t
		# Beta are observation probabilites
		# Also called the "emmission probabilities"
		# Beta[j][k]: probability of observing k in state j
		# Pi is the initial state distribution
		# Also called the "start probabilities"
		# Pi[i] is the probability of beginning in state i
		# Sum over all j in Alpha[i] must equal 1
		# Sum over all k in Beta[j] must equal 1
		# Sum over i in Pi must equal 1
		self.alpha = np.ones((states, states))
		self.beta = np.ones((states, observables))
		for i in range(0, states):
			self.alpha[i] = np.random.dirichlet(self.alpha[i])
			self.beta[i] = np.random.dirichlet(self.beta[i])
		self.pi = np.random.dirichlet([1] * states)

	# Decoding algorithms
	# The Viterbi decoding algorithm is used to find the most likely path of states
	# The standard Viterbi algorithm is below
	def decodeViterbi(self, obs):
		# Initialization
		T = len(obs)
		delta = np.zeros((self.numStates, T))
		path = np.zeros((self.numStates, T),'int32')

		# t = 0
		delta[:, 0] = np.squeeze(self.pi * self.beta[:, obs[0]])

		# Recursion
		# t > 0
		for t in range(1, T):
			delta[:, t] = np.max(np.dot(delta[:,[t-1]],[self.beta[:, obs[t]]])*self.alpha,0)
			path[:, t] = np.argmax(np.tile(delta[:,[t-1]],(1,self.numStates))*self.alpha,0)

		# Termination
		backtrace = [np.argmax(delta[:,-1])] # -1 addresses column T-1
		probability = np.max(delta[:,-1])

		# Backtrace
		for i in range(T-1, 0, -1):
			backtrace.append(path[backtrace[-1], i])
		# We need to move the first element to the end of backtrace
		# since it is actually the T'th element
		return backtrace[::-1], probability

	# The ViterbiLog algorithm is very similar to the standard algorithm
	# All of the computations are done in the log domain to avoid underflow
	def decodeViterbiLog(self, obs):
		# Initialization
		pi = np.log(self.pi)
		beta = np.log(self.beta)
		alpha = np.log(self.alpha)

		T = len(obs)
		delta = np.zeros((self.numStates, T))
		path = np.zeros((self.numStates, T),'int32')

		# t = 0
		delta[:, 0] = np.squeeze(pi + beta[:, obs[0]])

		# Recursion
		# t > 0
		for t in range(1, T):
			delta[:, t] = np.max(np.tile(delta[:,[t-1]],(1,self.numStates))+alpha,0) + beta[:, obs[t]]
			path[:, t] = np.argmax(np.tile(delta[:,[t-1]],(1,self.numStates))+alpha,0)

		# Termination
		backtrace = [np.argmax(delta[:,-1])] # -1 addresses column T-1
		probability = np.max(delta[:,-1])

		# Backtrace
		for i in range(T-1, 0, -1):
			backtrace.append(path[backtrace[-1], i])
		# We need to move the first element to the end of backtrace
		# since it is actually the T'th element
		return backtrace[::-1], probability

	# The Forward-Backward decoder uses separate forward and backward steps to decode
	def decodeForwardBackward(self, obs):
		# Forward pass
		T = len(obs)
		for t in range(0, T):
			
		return None

	def forward(self, obs):
		T = len(obs)
		for 

	# Training algorithms
	# The Viterbi training (or extraction) algorithm uses the Viterbi decoder to find alpha and beta
	# This algorithm employs K-means clustering for its unsupervised learning
	# It is often called Segmental K-means for this reason
	# It is also called Gaussian training because Gaussian mixture models are used to update alpha and beta
	def trainViterbi(self, data, epochs = 10):
		prob = 0.0
		for e in range(0, epochs):
			self.randomizeProbabilities()

			while True:
				b, p = self.decodeViterbi(data)
				oldAlpha = self.alpha
				oldBeta = self.beta

				# TODO: Update alpha and beta parameters
				if (np.sum(np.abs(self.alpha-oldAlpha))+np.sum(np.abs(self.beta-oldBeta))) < 0.00001:
					break

			if p > prob:
				newAlpha = self.alpha
				newBeta = self.beta
				prob = p
		self.alpha = newAlpha
		self.beta = newBeta

	# The Baum-Welch training algorithm uses several other algorithms
	# It is a combination of EM (Expectation-Maximation) applied to HMM
	# using the Forward-Backward steps to help update alpha and beta
	def trainBaumWelch(self, data):
		# not yet implemented
		return None
