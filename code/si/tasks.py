	
from tensorflow.examples.tutorials.mnist import input_data
from collections import namedtuple
import numpy as np

Task = namedtuple('Task', ['train', 'test'])
mnist = input_data.read_data_sets("/tmp/", one_hot=True)

def get_permuted_mnist(num_tasks):
		tasks = []
		for _ in range(num_tasks):
			task_permutation = np.random.permutation(784)
			# Apply task-specific permutation to training data
			permuted_train_images = mnist.train.images[:, task_permutation]
			permuted_train_labels = mnist.train.labels
			train_data = (permuted_train_images, permuted_train_labels)
			# Apply task-specific permutation to test data
			permuted_test_images = mnist.test.images[:, task_permutation]
			permuted_test_labels = mnist.test.labels
			test_data = (permuted_test_images, permuted_test_labels)
            # Create task        
			tasks.append(Task(train=train_data, test=test_data))
		return tasks


def get_split_mnist(num_tasks):

    def select_mnist_task(images, labels, digits):
        # Get indices of provided digits
        d1, d2 = digits
        mask = np.logical_or(np.argmax(labels, axis=1) == d1, np.argmax(labels, axis=1) == d2)
        images = images[mask]
        labels = labels[mask]
        # Convert one-hot to binary label [0, 1]
        labels = np.argmax(labels, axis=1)
        labels = np.array(labels == d2, dtype=np.float32)
        labels = np.eye(2)[labels.astype(np.int32)]
        return images, labels

    # Split MNIST task generation
    split_mnist_tasks = [(2 * i, 2 * i + 1) for i in range(num_tasks)]  # Create pairs of digits for each task
    tasks = []

    for digits in split_mnist_tasks:
        # Filter the dataset for the current task
        train_images, train_labels = select_mnist_task(mnist.train.images, mnist.train.labels, digits)
        test_images, test_labels = select_mnist_task(mnist.test.images, mnist.test.labels, digits)
        train_data = (train_images, train_labels)
        test_data = (test_images, test_labels)
        tasks.append(Task(train=train_data, test=test_data))

    return tasks
