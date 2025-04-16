import tensorflow as tf
import numpy as np

from alg import SI
from seed import seed_everything
from tasks import get_permuted_mnist, get_split_mnist
from nn import create_single_head_mlp, create_multi_head_mlp
from pathlib import Path
from utils import parse_args

def experiment(benchmark: str, seed: int, experiment_dir: Path, objective: str = "classification"):
	seed_everything(seed)
	if benchmark == "permuted_mnist":
		num_tasks = 10
		num_epochs = 20
		batch_size = 256
		lr = 0.001
		hidden_dim = 200
		# Parameters for SI
		si_lambda = 0.5
		tasks = get_permuted_mnist(num_tasks)
	else: 
		num_tasks = 5
		num_epochs = 120
		batch_size = 60000
		lr = 0.001
		hidden_dim = 256
		# Parameters for SI
		si_lambda = 1.0
		tasks = get_split_mnist(num_tasks)

	# Placeholders for x and y
	x = tf.placeholder(tf.float32, shape=[None, 784])
	if benchmark == "permuted_mnist":
		y_gt = tf.placeholder(tf.float32, shape=[None, 10])
	else:
		y_gt = tf.placeholder(tf.float32, shape=[None, 2])
	# Placeholders for task selection
	task_idx_ph = tf.placeholder(tf.int32, shape=[])
	# Neural network graph and weights
	if benchmark == "permuted_mnist":
		[y, weights] = create_single_head_mlp(x, hidden_dim)
	else:
		[y, weights] = create_multi_head_mlp(x, task_idx_ph, num_tasks, hidden_dim)

	si = SI(weights)
	# Loss and accuracy
	if objective == "classification":
		loss = tf.reduce_sum(
			tf.keras.backend.categorical_crossentropy(y_gt, y)
		) # If we use reduce_mean much worse ...
	else:
		loss = tf.reduce_sum(
			tf.reduce_sum(
				(y_gt - y) ** 2,
				axis=-1
			)
		) # If we use reduce_mean much worse ...
	accuracy = tf.reduce_mean(
		tf.cast(
			tf.equal(tf.argmax(y, axis=1), tf.argmax(y_gt, axis=1)),
			tf.float32
		)
	)
	# Create optimizer
	optimizer = tf.train.AdamOptimizer(lr)
	# Gradient of the loss function for the current task
	task_only_grads = optimizer.compute_gradients(loss, weights)
	# Gradient of the loss+penalty function, in order to both perform and to update w
	task_and_penalty_grads = optimizer.compute_gradients(loss + si_lambda*si.penalty(), weights)
	# Do single batch update graph
	train_single_batch = optimizer.apply_gradients(task_and_penalty_grads)
	update_w = si.update_w(task_only_grads, task_and_penalty_grads, lr)
	## Initialize session
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	# Train and eval
	history = []
	for task_idx, task in enumerate(tasks):
			print("[task={}]".format(task_idx + 1))
			x_train, y_train = task.train
			batch_size = min(batch_size, len(x_train))
			num_batches = int(len(x_train) / batch_size)
			# Train on the task
			for epoch in range(num_epochs):
				shuffle_idx = np.random.permutation(len(x_train))
				x_train_shuffled = x_train[shuffle_idx]
				y_train_shuffled = y_train[shuffle_idx]
				print("[task={}, epoch={}]".format(task_idx + 1, epoch + 1))
				for batch_idx in range(num_batches):
					start_idx = batch_idx * batch_size
					end_idx = start_idx + batch_size
					if benchmark == "permuted_mnist":
						sess.run(
							[train_single_batch, update_w],
							feed_dict={
								x: x_train_shuffled[start_idx:end_idx],
								y_gt: y_train_shuffled[start_idx:end_idx]
							}
						)
					else:
						sess.run(
							[train_single_batch, update_w],
							feed_dict={
								x: x_train_shuffled[start_idx:end_idx],
								y_gt: y_train_shuffled[start_idx:end_idx],
								task_idx_ph: task_idx
							}
						)
			# Update SI
			sess.run(si.update_omega())
			sess.run(si.store_previous_weights())
			sess.run(si.reset_w())
			# Compute per task metric
			task_metrics = []
			for tast_task_idx in range(task_idx + 1):
					x_test, y_test = tasks[tast_task_idx].test

					if objective == "classification":
						metric = accuracy
					else:
						metric = tf.sqrt(
							tf.reduce_mean(
								tf.reduce_sum(
									(y_gt - y) ** 2,
									axis=-1
								)
							)
						)
					if benchmark == "permuted_mnist":
						task_metric = sess.run(metric, feed_dict={x: x_test, y_gt: y_test})
					else:
						task_metric = sess.run(metric, feed_dict={x: x_test, y_gt: y_test, task_idx_ph: tast_task_idx})
					task_metrics.append(task_metric)
			history.append(task_metrics)
			print(np.array(task_metrics))
			print(np.array(task_metrics).mean())
			print("----")

	output_folder = experiment_dir / benchmark / "seed={}".format(seed)
	output_folder.mkdir(exist_ok=True, parents=True)
	np.save(str(output_folder / "history.npy"), history)

if __name__ == "__main__":
	# pmnist 17 works
	args = parse_args()
	print("Benchmark:", args.benchmark)
	print("Seed:", args.seed)
	print("Experiment Directory:", args.experiment_dir)
	print("Objective:", args.objective)
	experiment(
		args.benchmark,
		args.seed,
		args.experiment_dir,
		args.objective
	)