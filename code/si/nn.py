import tensorflow as tf
import numpy as np

def create_weight(input_size, output_size, init):
	if init == "normal":
		return tf.Variable(tf.random_normal([input_size, output_size], 0, 0.1))
	if init == "xavier":
		return tf.Variable(
			tf.random_uniform(
				[input_size, output_size],
				-6.0/np.sqrt(input_size + output_size), 6.0/np.sqrt(input_size + output_size)
			)
		)
	if init == "lecun":
		return tf.Variable(
			tf.random_uniform(
				[input_size, output_size],
				-1.0 / np.sqrt(input_size),  1.0 / np.sqrt(input_size)
			)
		)	

def create_bias(size):
	return tf.Variable(tf.zeros([size]))

def create_single_head_mlp(x, hidden_dim):
	# First hidden layer
	W1 = create_weight(784, hidden_dim, "lecun")
	b1 = create_bias(hidden_dim)
	# Second hidden layer
	W2 = create_weight(hidden_dim, hidden_dim, "lecun")
	b2 = create_bias(hidden_dim)
	# Head
	Wh = create_weight(hidden_dim, 10, "lecun")
	bh = create_bias(10)
	# Forward pass
	h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
	h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
	# Classify
	y = tf.nn.softmax(tf.matmul(h2, Wh) + bh, dim=-1)
	return y, [W1, b1, W2, b2, Wh, bh]

def create_multi_head_mlp(x_ph, task_idx_ph, num_heads, hidden_dim):
	# First hidden layer
	W1 = create_weight(784, hidden_dim, "normal")
	b1 = create_bias(hidden_dim)
	# Second hidden layer
	W2 = create_weight(hidden_dim, hidden_dim, "normal")
	b2 = create_bias(hidden_dim)
	# Create head per task
	Whs = []
	bhs = []
	for _ in range(num_heads):
		Whs.append(create_weight(hidden_dim, 2, "normal"))
		bhs.append(create_bias(2))
	# Forward pass through shared layers
	h1 = tf.nn.relu(tf.matmul(x_ph, W1) + b1)
	h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
	# Push through the heads
	all_outputs = []
	for head_idx in range(num_heads):
		head_output = tf.nn.softmax(tf.matmul(h2, Whs[head_idx]) + bhs[head_idx], dim=-1)
		all_outputs.append(head_output)
	# Select the appropriate head
	y = tf.case({
		tf.equal(task_idx_ph, i): lambda i=i: all_outputs[i] for i in range(num_heads)},
		default=lambda: all_outputs[0],
		exclusive=True
	)
	return y, ([W1, b1, W2, b2] + Whs + bhs)
