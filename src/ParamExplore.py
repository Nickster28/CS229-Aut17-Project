import tensorflow as tf
from data.DataLoader import DataLoader
from ClassificationNNModel import ClassificationNNModel
from random import uniform
from RunNN import Config

if __name__ == "__main__":
	config = Config("h", 10, "classification", 5)

	results = []
	for i in range(100):
		print("Iteration %i" % i)
		config.lr = 10**uniform(-2, -6)

		with tf.Graph().as_default():
			model = ClassificationNNModel(config)
			loader = DataLoader()
			loader.loadData("../dataset/dataset-train.npy", filterFn=lambda x: x.getHelpTimeMinutes() >= 2.0, log=False)
			model.run(loader, "h", train=True, log=False)
			loss = model.run(loader, "h", train=False, log=False)
			results.append((config.lr, loss))

	print(sorted(results, key=lambda r: r[1]))