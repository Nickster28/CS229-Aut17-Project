"""
Neural Model Runner

# HOW TO RUN
python RunNN.py [train/dev/test] --model=[c/r] --time=[w/h/t]
"""
import argparse
import tensorflow as tf
from Vectorizers import init_vectorizers
from data.DataLoader import DataLoader
from Vectorizers import init_vectorizers
from data.parser import LaIRRequest
from RegressionNNModel import RegressionNNModel
from ClassificationNNModel import ClassificationNNModel
from ClassificationNNModel_newcost import ClassificationNNModel as NewClassification

class Config:
	"""Holds model hyperparams and data information.

	The config class is used to store various hyperparameters and dataset
	information parameters. Model objects are passed a Config() object at
	instantiation.
	"""
	dropout = 1 # TODO: re-add dropout
	n_features = 5484
	hidden_size = 1000
	n_hidden_layers = 5
	batch_size = 25
	regularization = 0.1
	lr = 0.005

	def __init__(self, params):
		if params.model == "c":
			self.output_path = "NNResults/classification/%s/%s/" % (params.time, params.buckets)
		else:
			self.output_path = "NNResults/regression/%s/" % params.time

		self.model_output = self.output_path + "model.weights"
		self.n_epochs = params.epochs if 'epochs' in params else None
		self.bucketString = params.buckets
		self.lr = params.lr

def createModel(params):
	"""Returns the model depending on the specified model type (either 'r' or 'c')"""
	config = Config(params)
	if params.model == 'r':
		return RegressionNNModel(config)
	elif params.model == 'c':
		return ClassificationNNModel(config)
	else:
		return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Trains and evaluates ClassificationNNModel')
	subparsers = parser.add_subparsers()

	command_parser = subparsers.add_parser('train', help='')
	command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="../dataset/dataset-train.npy", help="Train CSV")
	command_parser.add_argument('-t', '--time', type=str)
	command_parser.add_argument('-m', '--model', type=str)
	command_parser.add_argument('-b', '--buckets', type=str, default="24w")
	command_parser.add_argument('-e', '--epochs', type=int, default=100)
	command_parser.add_argument('-l', '--lr', type=float, default=0.01)
	command_parser.set_defaults(func=lambda: 'train')

	command_parser = subparsers.add_parser('dev', help='')
	command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="../dataset/dataset-dev.npy", help="Dev CSV")
	command_parser.add_argument('-t', '--time', type=str)
	command_parser.add_argument('-m', '--model', type=str)
	command_parser.add_argument('-b', '--buckets', type=str, default="24w")
	command_parser.add_argument('-l', '--lr', type=float, default=0.01)
	command_parser.set_defaults(func=lambda: 'dev')

	command_parser = subparsers.add_parser('test', help='')
	command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="../dataset/dataset-test.npy", help="Test CSV")
	command_parser.add_argument('-t', '--time', type=str)
	command_parser.add_argument('-m', '--model', type=str)
	command_parser.add_argument('-b', '--buckets', type=str, default="24w")
	command_parser.add_argument('-l', '--lr', type=float, default=0.01)
	command_parser.set_defaults(func=lambda: 'test')

	ARGS = parser.parse_args()

	if 'func' not in ARGS or ARGS.func is None:
		parser.print_help()
	elif ARGS.time not in ['w', 'h', 't']:
		print("ERROR: invalid time '%s'" % ARGS.time)
	else:
		with tf.Graph().as_default():

			model = createModel(ARGS)
			loader = DataLoader()
			vectorizers = init_vectorizers()

			# Filter out bad requests if we are training on help time
			if ARGS.time == 'h':
				loader.loadData(ARGS.data.name, filterFn=lambda x: x.getHelpTimeMinutes() >= 2.0)
			else:
				loader.loadData(ARGS.data.name)

			# Training
			if ARGS.func() == 'train':
				model.run(loader, vectorizers, ARGS.time, run_type='train')
				train_loss = model.run(loader,vectorizers, ARGS.time, run_type='dev')
				print("Train accuracy = %f" % (1 - train_loss))

			# Dev / Test
			else:
				output = model.run(loader, vectorizers, ARGS.time, run_type=ARGS.func())
				print("Evaluation accuracy = %f" % (1 - output))
			
