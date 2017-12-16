from data.DataLoader import DataLoader
from data.parser import LaIRRequest
import csv
import collections
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

"""
CLARIFICATION

A Vectorizer is NOT the same thing as a feature extractor. A vectorizer
takes an element in a request that is either non-numerical (i.e. the help
request) or has an arbitrary numerical value (e.g. student ID). In that case,
a vectorizer is used to produce vectorial representation of these elements.
"""
def init_vectorizers(log=True):
	"""
	Initializes vectorizers.
	"""
	if log: print("Initializing vectorizers...", end="\r")

	# Create DataLoaders for train and full for the vectorizers
	trainD = DataLoader()
	trainD.loadData('../dataset/dataset-train.npy')
	fullD = DataLoader()
	fullD.loadData('../dataset/dataset.npy')

	# Create the vectorizers
	return [
		TFIDFRequestTextVectorizer(trainD),
		HelperIDVectorizer(fullD),
		CourseIDVectorizer(),
		RequestTimeVectorizer(),
		StudentVectorizer(fullD),
		PastRequestsVectorizer(fullD),
		DueDateVectorizer()
	]

class BaseVectorizer:
	"""
	Base Vectorizer class. Provides a basic constructor implementation
	(which can be overriddden in vectorizers subclassed from this) and
	several abstract methods that SubVectorizers must implement.
	"""
	def __getitem__(self, key):
		"""
		Override __getitem__ to - given an element you wish to vectorize -
		return a vector representing the element.
		"""
		raise NotImplementedError("Getitem must be overridden in subclass")

	def feature_names(self):
		"""
		Returns list of names of features that correspond to elements in the vector.
		Useful for debugging.
		"""
		raise NotImplementedError("feature_names must be overidden in subclass")

class PastRequestsVectorizer(BaseVectorizer):
	"""
	Converts request to information about all past requests that day,
	as well as who is currently in the queue.
	"""
	def __init__(self, loader):
		self.loader = loader

	def __getitem__(self, request):
		total_wait_time_reqs = 0
		total_help_time_reqs = 0
		total_wait_time = 0
		total_help_time = 0
		enqueued_requests = 0
		for dbRequest in self.loader.laIRRequests:
			if dbRequest.claimDateTime < request.requestDateTime and dbRequest.claimDateTime.day == request.requestDateTime.day:
				total_wait_time_reqs += 1
				total_wait_time += dbRequest.getWaitTimeMinutes()
			if dbRequest.closeDateTime < request.requestDateTime and dbRequest.closeDateTime.day == request.requestDateTime.day and dbRequest.getHelpTimeMinutes() >= 2.0:
				total_help_time_reqs += 1
				total_help_time += dbRequest.getHelpTimeMinutes()
			if dbRequest.claimDateTime.day == request.claimDateTime.day and dbRequest.claimDateTime > request.requestDateTime and dbRequest.requestDateTime < request.requestDateTime:
				enqueued_requests += 1

		return np.array([ total_wait_time_reqs, total_wait_time, total_help_time_reqs, total_help_time, enqueued_requests ])


class TFIDFRequestTextVectorizer(BaseVectorizer):
	"""
	Converts Help request text to vector representation
	"""
	def __init__(self, loader):
		corpus = loader.laIRRequestDescriptions
		self.vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer = 'word', stop_words = 'english')
		self.vectorizer.fit(corpus)

	def feature_names(self):
		return self.vectorizer.get_feature_names()

	def __getitem__(self, request):
		text = request.problemDescription
		textVec = self.vectorizer.transform([text]).toarray()
		output = np.concatenate((textVec, np.full((1, 1), len(text))), axis=1)
		return output

class CourseIDVectorizer(BaseVectorizer):
	"""
	Course IDs to vectors. Each course is a different element.
	"""
	def __init__(self):
		self.vectors = {
			"A": np.array([1, 0, 0]),
			"B": np.array([0, 1, 0]),
			"X": np.array([0, 0, 1])
		}

	def feature_names(self):
		return self.vectors.keys()

	def __getitem__(self, request):
		letter = request.courseLetter
		return np.array(self.vectors[letter])

class HelperIDVectorizer(BaseVectorizer):
	"""
	Given a list of helpers currently on shift, the vector
	representation is a vector with as many elements as there are SLs
	represented in the dataset, with 1s in the on-shift SL's corresponding
	elems, and 0 everywhere else.
	"""
	def __init__(self, loader):
		self.feature_names = loader.laIRHelpers

	def feature_names(self):
		return self.feature_names

	def __getitem__(self, request):
		request_helpers = request.helperIds
		return np.array([[0 if helper in request_helpers else 1 for helper in self.feature_names]])

class RequestTimeVectorizer(BaseVectorizer):
	"""
	Vectorize current quarter and current day of week using one-hot
	representations with 4 and 7 elements respectively; also make polynomial features
	for hour offset into the LaIR up to order 4.
	"""
	def __init__(self):
		self.quarters = {
			"Autumn": np.array([1, 0, 0, 0]),
			"Winter": np.array([0, 1, 0, 0]),
			"Spring": np.array([0, 0, 1, 0]),
			"Summer": np.array([0, 0, 0, 1])
		}

	def __getitem__(self, request):
		dt = request.requestDateTime

		weekday_feature = np.zeros(7)
		weekday_feature[dt.weekday()] = 1

		hour_offset = dt.hour - 18.0
		hour_feature = np.array([ hour_offset ** order for order in range(4) ])

		quarter_feature = self.quarters[request.quarter]

		return np.concatenate([weekday_feature, hour_feature, quarter_feature])

class StudentVectorizer(BaseVectorizer):
	"""
	Vectorize the number of times a student has been to the LaIR, as well as the
	total help time of that student.
	"""
	def __init__(self, loader):
		self.loader = loader

	def __getitem__(self, request):
		total_reqs = 0
		total_help_time = 0
		for dbRequest in self.loader.laIRRequests:
			if dbRequest.studentId == request.studentId and dbRequest.requestDateTime < request.requestDateTime and dbRequest.getHelpTimeMinutes() >= 2.0:
				total_reqs += 1
				total_help_time += dbRequest.getHelpTimeMinutes()

		return np.array([ total_reqs, total_help_time ])

class DueDateVectorizer(BaseVectorizer):
	"""
	Vectorize the number of days to the next due date for all CS106 classes.
	"""
	def __init__(self):
		self.due_dates = {
			('2017', 'Autumn', 'A') : [(10, 6, 2017), (10, 16, 2017), (10, 25, 2017), (11, 6, 2017), (11, 15, 2017), (11, 29, 2017), (12, 8, 2017)],
			('2017', 'Autumn', 'B') : [(10, 2, 2017), (10, 9, 2017), (10, 18, 2017), (10, 25, 2017), (11, 6, 2017), (11, 13, 2017), (11, 27, 2017), (12, 6, 2017)],
			('2017', 'Autumn', 'X') : [(10, 6, 2017), (10, 13, 2017), (10, 20, 2017), (10, 30, 2017), (11, 10, 2017), (11, 17, 2017), (11, 29, 2017), (12, 8, 2017)],

			('2017', 'Spring', 'A') : [(4, 14, 2017), (4, 24, 2017), (5, 1, 2017), (5, 8, 2017), (5, 17, 2017), (5, 26, 2017), (6, 5, 2017)],
			('2017', 'Spring', 'B') : [(4, 13, 2017), (4, 20, 2017), (4, 27, 2017), (5, 11, 2017), (5, 18, 2017), (5, 25, 2017), (6, 6, 2017)],
			('2017', 'Spring', 'X') : [(1, 1, 1)],

			('2017', 'Winter', 'A') : [(1, 20, 2017), (1, 30, 2017), (2, 8, 2017), (2, 17, 2017), (2, 27, 2017), (3, 8, 2017), (3, 17, 2017)],
			('2017', 'Winter', 'B') : [(1, 13, 2017), (1, 23, 2017), (1, 30, 2017), (2, 6, 2017), (2, 17, 2017), (3, 3, 2017), (3, 10, 2017), (3, 17, 2017)],
			('2017', 'Winter', 'X') : [(1, 13, 2017), (1, 11, 2017), (1, 20, 2017), (1, 27, 2017), (2, 17, 2017), (3, 3, 2017), (3, 10, 2017), (3, 17, 2017)],

			('2016', 'Autumn', 'A') : [(10, 7, 2016), (10, 17, 2016), (10, 26, 2016), (11, 7, 2016), (11, 16, 2016), (11, 30, 2016), (12, 9, 2016)],
			('2016', 'Autumn', 'B') : [(10, 7, 2016), (10, 15, 2016), (10, 24, 2016), (11, 8, 2016), (11, 17, 2016), (11, 30, 2016)],
			('2016', 'Autumn', 'X') : [(10, 7, 2016), (10, 17, 2016), (10, 26, 2016), (11, 4, 2016), (11, 14, 2016), (11, 28, 2016), (11, 9, 2016)],

			('2016', 'Spring', 'A') : [(4, 8, 2016), (4, 18, 2016), (4, 27, 2016), (5, 6, 2016), (5, 16, 2016), (5, 23, 2016), (6, 1, 2016)],
			('2016', 'Spring', 'B') : [(4, 1, 2016), (4, 15, 2016), (4, 22, 2016), (4, 29, 2016), (5, 11, 2016), (5, 20, 2016), (5, 30, 2016)],
			('2016', 'Spring', 'X') : [(1, 1, 1)],

			('2016', 'Winter', 'A') : [(1, 15, 2016), (1, 22, 2016), (2, 3, 2016), (2, 17, 2016), (2, 29, 2016), (2, 11, 2016)],
			('2016', 'Winter', 'B') : [(1, 15, 2016), (1, 25, 2016), (2, 3, 2016), (2, 15, 2016), (2, 24, 2016), (3, 2, 2016), (3, 11, 2016)],
			('2016', 'Winter', 'X') : [(1, 15, 2016), (1, 25, 2016), (2, 3, 2016), (2, 8, 2016), (2, 19, 2016), (2, 29, 2016), (3, 9, 2016), (3, 18, 2016)],

			('2015', 'Autumn', 'A') : [(10, 2, 2015), (10, 12, 2015), (10, 21, 2015), (11, 2, 2015), (11, 11, 2015), (11, 20, 2015), (12, 4, 2015)],
			('2015', 'Autumn', 'B') : [(10, 2, 2015), (10, 10, 2015), (10, 19, 2015), (10, 30, 2015), (11, 9, 2015), (11, 18, 2015), (12, 2, 2015)],
			('2015', 'Autumn', 'X') : [(10, 2, 2015), (10, 9, 2015), (10, 16, 2015), (10, 21, 2015), (11, 4, 2015), (11, 11, 2015), (11, 18, 2015), (12, 4, 2015)],

			('2015', 'Spring', 'A') : [(4, 10, 2015), (4, 20, 2015), (4, 29, 2015), (5, 8, 2015), (5, 18, 2015), (5, 27, 2015), (6, 2, 2015)],
			('2015', 'Spring', 'B') : [(4, 10, 2015), (4, 17, 2015), (4, 24, 2015), (5, 4, 2015), (5, 14, 2015), (5, 22, 2015), (6, 1, 2015)],
			('2015', 'Spring', 'X') : [(1, 1, 1)],

			('2015', 'Winter', 'A') : [(1, 16, 2015), (1, 18, 2015), (1, 26, 2015), (2, 2, 2015), (2, 9, 2015), (2, 18, 2015), (2, 27, 2015), (3, 9, 2015), (3, 17, 2015)],
			('2015', 'Winter', 'B') : [(1, 16, 2015), (1, 26, 2015), (2, 4, 2015), (2, 6, 2015), (2, 13, 2015), (2, 23, 2015), (3, 4, 2015), (3, 13, 2015)],
			('2015', 'Winter', 'X') : [(1, 16, 2015), (1, 26, 2015), (2, 4, 2015), (2, 9, 2015), (2, 20, 2015), (3, 2, 2015), (3, 9, 2015), (3, 17, 2015)],
		}

		self.due_dates = { key : [ datetime(date[2], date[0], date[1]) for date in dates ] for key, dates in self.due_dates.items() }

	def __getitem__(self, request):
		def days_until_next_date(date, due_dates):
			for due_date in due_dates:
				diff = (due_date - date).total_seconds() / 86400.0
				if diff > 0:
					return diff
			return -1

		naive_date = datetime(request.requestDateTime.year, request.requestDateTime.month, request.requestDateTime.day)
		daysToA = days_until_next_date(naive_date, self.due_dates[(str(request.claimDateTime.year), request.quarter, 'A')])
		daysToB = days_until_next_date(naive_date, self.due_dates[(str(request.claimDateTime.year), request.quarter, 'B')])
		daysToX = days_until_next_date(naive_date, self.due_dates[(str(request.claimDateTime.year), request.quarter, 'X')])
		return np.array([daysToA, daysToB, daysToX])

