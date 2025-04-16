import tensorflow as tf

class SI:
	def __init__(self, weights):
		self.weights = weights # current weights
		self.previous_weights = {
			weight.op.name: tf.Variable(
				tf.zeros(weight.get_shape()),
				trainable=False
			)
			for weight in weights
		} # previous task optimal weights
		self.omega = {
			weight.op.name: tf.Variable(
				tf.zeros(weight.get_shape()),
				trainable=False
			)
			for weight in weights		
		} # omega variables from the paper
		self.w = {
			weight.op.name: tf.Variable(
				tf.zeros(weight.get_shape()),
				trainable=False
			)
			for weight in weights		
		} # w variables from the paper
		self.eps = 0.1

	def penalty(self):
		return tf.add_n([
			tf.reduce_sum(
				self.omega[w.op.name] * ((self.previous_weights[w.op.name] - w)**2)
			)
			for w in self.weights
		])
	
	def update_w(self, task_only_grads, task_and_penalty_grads, lr):
		return tf.group(*[
			tf.assign_add(
				self.w[weight.op.name],
				lr*task_and_penalty_grads[param_idx][0]*task_only_grads[param_idx][0]
			)
			for param_idx, (_, weight) in enumerate(task_and_penalty_grads)
		])
	
	def update_omega(self):
		return tf.group(*[
			tf.assign_add(
				self.omega[weight.op.name],
				(self.w[weight.op.name] / (self.eps + (weight - self.previous_weights[weight.op.name]) ** 2 ))
			)
			for weight in self.weights
		])
	
	def reset_w(self):
		return tf.group(*[
			tf.assign(
				self.w[weight.op.name],
				tf.zeros_like(self.w[weight.op.name])
			)
			for weight in self.weights
		])
	
	def store_previous_weights(self):
		return tf.group(*[
			tf.assign(self.previous_weights[weight.op.name], weight)
			for weight in self.weights
		])
