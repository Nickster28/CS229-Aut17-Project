from Vectorizers import *
from data.DataLoader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from util import make_buckets, make_bucket_mapper
import argparse

def trainModel(ModelType, X, y):
	if ModelType == SGDRegressor:
		model = SGDRegressor(loss='epsilon_insensitive', max_iter=100)
	else:
		model = ModelType()

	model.fit(X, y)
	accuracy = model.score(X, y)
	print("Model training score: {}".format(accuracy))
	return model

def evaluateModel(model, X, y):
	accuracy = model.score(X, y)
	print("Model evaluation score: {}".format(accuracy))

	if type(model) == SGDRegressor:
		pred = model.predict(X)
		print("Mean squared error: {}".format(mean_squared_error(y, pred)))

def runLogistic(args):
	run(LogisticRegression, args)

def runLinear(args):
	run(SGDRegressor, args)

def run(ModelType, args):
	print("\n********* %s %s Model *********" % (("Logistic" if ModelType == LogisticRegression else "Linear"), ("Wait" if args.time == 'w' else "Help")))
	vectorizers = init_vectorizers()
	trainLoader = DataLoader()
	evaluateLoader = DataLoader()
	testLoader = DataLoader()

	# Filter out bad requests if we are running on help time
	if args.time == 'h':
		trainLoader.loadData('../dataset/dataset-train.npy', filterFn=lambda x: x.getHelpTimeMinutes() >= 2.0)
		evaluateLoader.loadData('../dataset/dataset-dev.npy', filterFn=lambda x: x.getHelpTimeMinutes() >= 2.0)
		testLoader.loadData('../dataset/dataset-test.npy', filterFn=lambda x: x.getHelpTimeMinutes() >= 2.0)
	else:
		trainLoader.loadData('../dataset/dataset-train.npy')
		evaluateLoader.loadData('../dataset/dataset-dev.npy')
		testLoader.loadData('../dataset/dataset-test.npy')

	if ModelType == LogisticRegression:
		buckets = make_buckets(trainLoader, args.buckets, args.time)
		mapper = make_bucket_mapper(buckets)
	else:
		mapper = lambda x: x

	labelFn = lambda x: mapper(x.getWaitTimeMinutes() if args.time == 'w' else x.getHelpTimeMinutes())
	trainLabels = trainLoader.getLabels(labelFn)
	trainInputs = trainLoader.applyVectorizers(vectorizers, "train", args.time)
	devLabels = evaluateLoader.getLabels(labelFn)
	devInputs = evaluateLoader.applyVectorizers(vectorizers, "dev", args.time)
	testLabels = testLoader.getLabels(labelFn)
	testInputs = evaluateLoader.applyVectorizers(vectorizers, "test", args.time)

	trainedModel = trainModel(ModelType, trainInputs, trainLabels)
	evaluateModel(trainedModel, devInputs, devLabels)
	evaluateModel(trainedModel, testInputs, testLabels)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Trains and evaluates Non-NN Models')
	subparsers = parser.add_subparsers()

	command_parser = subparsers.add_parser('logistic', help='')
	command_parser.add_argument('-t', '--time', type=str)
	command_parser.add_argument('-b', '--buckets', type=str, default="24w")
	command_parser.set_defaults(func=runLogistic)

	command_parser = subparsers.add_parser('linear', help='')
	command_parser.add_argument('-t', '--time', type=str)
	command_parser.set_defaults(func=runLinear)

	ARGS = parser.parse_args()

	if 'func' not in ARGS or ARGS.func is None:
		parser.print_help()
	elif ARGS.time not in ['w', 'h', 't']:
		print("ERROR: invalid time '%s'" % ARGS.time)
	else:
		ARGS.func(ARGS)
