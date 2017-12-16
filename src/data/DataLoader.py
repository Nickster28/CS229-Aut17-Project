from .parser import LaIRRequest
from tqdm import tqdm
import numpy as np
import os

VECTORIZED_FILENAME = "Vectorized.npy"

'''
CLASS: DataLoader
-----------------
A class responsible for reading in the parsed dataset from .npy and storing it
as a list of LaIRRequest objects for ease of training/testing.

A DataLoader is initially empty; to fill it with the contents of a .npy file
(that MUST HAVE been previously outputted by parser.py), call the loadData
method and pass in a filename.  While loading the data, the DataLoader will
initialize all internal state.  Once you call loadData, the internal DataLoader
state *cannot be changed* further (i.e. subsequent calls to loadData are no-ops).

A DataLoader has the following instance variables:
	- laIRRequests: a list of LaIRRequest objects read in from file
	- laIRHelpers: a list of all helper IDs who were on shift across all requests from file
	- lairRequestDescriptions: a list of all request descriptions read in from file
-----------------
'''
class DataLoader:

	'''
	METHOD: __init__
	----------------
	Parameters: NA
	Returns: NA

	Initializes all private instance variables to None so we know when we have/
	haven't loaded data in.
	----------------
	'''
	def __init__(self):
		self.laIRRequests = None
		self.laIRHelpers = None
		self.laIRRequestDescriptions = None

	'''
	METHOD: loadData
	----------------
	Parameters:
		filename - the name of the CSV file to load in (MUST have been created
					by the parser.py script)

	Returns: NA

	Initializes all internal DataLoader state by reading in the given file.
	If our state has already been initialized, does nothing.
	----------------
	'''
	def loadData(self, filename, filterFn=None, log=True):
		if self.laIRRequests is None:
			self.laIRRequests = np.load(filename)
			if filterFn is not None:
				self.laIRRequests = list(filter(filterFn, self.laIRRequests))
				if log: print("Filtered to %i requests" % len(self.laIRRequests))

			# Calculate lairHelpers and lairRequestDescriptions
			helpersSet = set()
			descriptions = []
			for request in self.laIRRequests:
				helpersSet |= set(request.helperIds)
				descriptions.append(request.problemDescription)

			self.laIRHelpers = list(helpersSet)
			self.laIRRequestDescriptions = descriptions

	'''
	METHOD: applyVectorizers
	------------------------
	Parameters:
		vectorizers - a list of vectorizer functions that each take in a single
						lair request and return its vectorized representation
		run_type - either 'train', 'dev' or 'test'
		timeChar - either 'w' (wait) or 'h' (help)
		log=True - whether or not to log more verbose output to the console
	------------------------
	'''
	def applyVectorizers(self, vectorizers, run_type, timeChar, log=True):
		if len(vectorizers) == 0:
			return np.array([])

		# Try to load cached vectorizers
		cached_filename = "%s-%s-%s" % (run_type, timeChar, VECTORIZED_FILENAME)
		if os.path.isfile(cached_filename):
			if log: print('Loading vectorizers from file')
			return np.load(cached_filename)
		else:

			# Generate vectorized input from scratch
			if log: print('Applying vectorizers...')
			X = []
			iterObj = tqdm(self.laIRRequests) if log else self.laIRRequests
			for request in iterObj:
				features = []
				for vectorizer in vectorizers:
					features.append(vectorizer[request].flatten())

				X.append(np.concatenate(features))

			if log: print('Applying vectorizers... Done!')
			X = np.array(X)
			np.save(cached_filename, X)
			return X

	'''
	METHOD: getLabels
	-----------------
	Parameters:
		mapFn - a function that takes in a lair request and returns its label

	Returns: a numpy array containing the label for each LaIR request.
	-----------------
	'''
	def getLabels(self, mapFn):
		return np.array(list(map(mapFn, self.laIRRequests)))
