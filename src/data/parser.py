'''
FILE: parser.py
---------------
Parser script for converting from the CS198-provided csv files:
	courses.csv
	help_requests.csv
	helper_assignments.csv
	helper_checkins.csv
	quarters.csv
	staff_relations.csv
	student_relations.csv

to one generated file per dataset type (60-20-20 split):
	train.npy
	dev.npy
	test.npy

where each row in the output contains all information for
a single LaIR request.  This script does this by combining
information across the provided csv files, and discarding
rows we do not want to use in our dataset (such as requests that were
mistakenly not closed, or requests that were reassigned).
---------------
'''

import csv
import dateutil.parser
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# Constants for names of input files
DATASET_DIR = "../../dataset/"
COURSES_FILENAME = DATASET_DIR + "courses.csv"
HELP_REQUESTS_FILENAME = DATASET_DIR + "help_requests.csv"
HELPER_ASSIGNMENTS_FILENAME = DATASET_DIR + "helper_assignments.csv"
HELPER_CHECKINS_FILENAME = DATASET_DIR + "helper_checkins.csv"
QUARTERS_FILENAME = DATASET_DIR + "quarters.csv"
STAFF_RELATIONS_FILENAME = DATASET_DIR + "staff_relations.csv"
STUDENT_RELATIONS_FILENAME = DATASET_DIR + "student_relations.csv"


'''
CLASS: LaIRRequest
------------------
This class represents all data about a single LaIR request.  It contains
the following instance variables:
	- courseLetter: either 'A', 'B', or 'X', course of requester
	- quarter: either 'Autumn', 'Winter' or 'Spring'
	- year: year number in which request was made
	- studentId: number identifying requester
	- problemDescription: string, requester's description of problem
	- requestDateTime: date/time of request creation
	- claimDateTime: date/time of request being claimed by helper
	- closeDateTime: date/time of request being closed by helper
	- helperIds: list of staff ids who were on duty when request was made

If you create an empty LaIRRequest and want to initialize it using relations
loaded from various CSV files, call the initializeFromRelations method and
provide the various relation objects corresponding to the request you would like
to represent.

To save a list of LaIRRequests to disk, you can call the class method
LaIRRequest.saveToFile() and pass in a filename and list of LaIRRequests.
That file will be saved as a .npy file that can be loaded using numpy.load.
------------------
'''
class LaIRRequest:

	'''
	CLASS METHOD: createLaIRRequests
	--------------------------------
	Parameters:
		cls - the Class object
		courses - map from id to Course
		help_requests - map from id to HelpRequest
		helper_assignments - map from help request ID to HelperAssignment
		helper_checkins - map from id to HelperCheckin
		quarters - map from id to Quarter
		staff_relations - map from person ID to StaffRelation
		student_relations - map from id to StudentRelation

	Returns: a list of LaIRRequests created from the provided parameters.
	--------------------------------
	'''
	@classmethod
	def createLaIRRequests(cls, courses, help_requests, helper_assignments, helper_checkins, quarters, staff_relations, student_relations):
		lairRequests = []
		print("Creating LaIR requests....")
		for requestId, request in tqdm(help_requests.items()):

			# Make sure there is a student relation and helper assignment for this request
			# There may not be if the student is not logged, or the request was handled by multiple helpers
			if request.student_relation_id in student_relations and requestId in helper_assignments:
				studentRelation = student_relations[request.student_relation_id]
				helperAssignment = helper_assignments[requestId]

				# Make sure this student is taking one of the 106 courses
				if studentRelation.course_id in courses:
					course = courses[studentRelation.course_id]
					quarter = quarters[course.quarter_id]
					lairRequest = LaIRRequest()
					lairRequest.initializeWithRelations(request, studentRelation, helperAssignment, course, quarter, helper_checkins)

					if (lairRequest.isValidRequest()):
						lairRequests.append(lairRequest)

		return lairRequests

	'''
	CLASS METHOD: saveToFile
	-----------------------
	Parameters:
		cls - the Class Object
		filename - the base name of the files to create
		requests - list of LaIRRequest objects to write to file
		partition=True - whether to also partition into train/dev/test datasets (60/20/20)

	Returns: NA

	Saves the given LaIR request array as a numpy object named FILENAME.npy.
	Each row in the numpy array is the LaIR request.  If partition=True, then
	a 60-20-20 split of the dataset is also saved as FILENAME-train.npy, 
	FILENAME-dev.npy and FILENAME-test.npy, respectively.
	-----------------------
	'''
	@classmethod
	def saveToFile(cls, filename, requests, partition=True):
		fullDataset = np.array(requests)
		np.random.shuffle(fullDataset)

		np.save(DATASET_DIR + filename, fullDataset)
			
		if partition:
			trainEnd = int(len(fullDataset) * 0.6)
			devEnd = int(len(fullDataset) * 0.8)
			splitResults = np.split(fullDataset, [trainEnd, devEnd])
			datasets = [("train", splitResults[0]), ("dev", splitResults[1]), ("test", splitResults[2])]

			for (datasetName, dataset) in datasets:
				np.save(DATASET_DIR + filename + "-" + datasetName, np.array(dataset))

	'''
	METHOD: __init__
	----------------
	Parameters: NA
	----------------
	'''
	def __init__(self, line=None):
		self.courseLetter = None
		self.quarter = None
		self.year = None
		self.studentId = None
		self.problemDescription = None
		self.requestDateTime = None
		self.claimDateTime = None
		self.closeDateTime = None
		self.helperIds = []

	'''
	METHOD: initializeWithRelations
	-------------------------------
	Parameters:
		helpRequest - the HelpRequest corresponding to this LaIR Request
		studentRelation - the StudentRelation corresponding to this LaIR Request
		helperAssignment - the HelperAssignment corresponding to this LaIR Request
		course - the Course corresponding to this LaIR request
		quarter - the Quarter corresponding to this LaIR request
		helper_checkins - the dict of all HelperCheckin objects for SL shifts, indexed by id.

	Returns: NA

	Fills in the current LaIRRequest object with the data given by the parameters.
	-------------------------------
	'''
	def initializeWithRelations(self, helpRequest, studentRelation, helperAssignment, course, quarter, helper_checkins):
			self.courseLetter = course.course_number[len("106"):].upper()
			self.quarter = quarter.quarter_name
			self.year = quarter.year
			self.studentId = studentRelation.person_id
			self.problemDescription = helpRequest.problem_description
			self.requestDateTime = helpRequest.request_time
			self.claimDateTime = helperAssignment.claim_time
			self.closeDateTime = helperAssignment.close_time

			for helper_checkin_id, helper_checkin in helper_checkins.items():
				if helper_checkin.check_in_time <= self.requestDateTime and helper_checkin.check_out_time >= self.requestDateTime:
					self.helperIds.append(helper_checkin.person_id)

	'''
	METHOD: isValidRequest
	----------------------
	Parameters: NA
	Returns: whether or not this is a valid LaIR Request.  See inline comments
	for explanation.
	----------------------
	'''
	def isValidRequest(self):
		requestHour = self.requestDateTime.hour
		requestDay = self.requestDateTime.day
		closeHour = self.closeDateTime.hour
		closeDay = self.closeDateTime.day

		requestNextDay = datetime(self.requestDateTime.year, 
			self.requestDateTime.month, 
			self.requestDateTime.day) + timedelta(days=1)

		closeDayDate = datetime(self.closeDateTime.year, self.closeDateTime.month, self.closeDateTime.day)

		# If this request came in before 5PM or was closed > 1 day later, it's invalid
		if requestHour < 17 or requestNextDay < closeDayDate:
			return False

		# If we get here, request was put in 5PM onwards AND resolved either same day or next day

		# If this request was resolved on the same day, it's valid
		if requestDay == closeDay:
			return True

		# If we get here, request was put in 5PM onwards AND resolved next day

		# If this request was resolved after 3AM, it's invalid
		if closeHour >= 3:
			return False

		# If we get here, request was put in 5PM onwards AND resolved next day by 3AM

		return True

	'''
	METHOD: getWaitTimeMinutes
	--------------------------
	Parameters: NA
	Returns: the wait time (in minutes) for this request
	--------------------------
	'''
	def getWaitTimeMinutes(self):
		return (self.claimDateTime - self.requestDateTime).total_seconds() / 60.0

	'''
	METHOD: getHelpTimeMinutes
	--------------------------
	Parameters: NA
	Returns: the help time (in minutes) for this request
	--------------------------
	'''
	def getHelpTimeMinutes(self):
		return (self.closeDateTime - self.claimDateTime).total_seconds() / 60.0

	'''
	METHOD: getTotalTimeMinutes
	---------------------------
	Parameters: NA
	Returns: the total time (wait time + help time) (in minutes) for this request
	'''
	def getTotalTimeMinutes(self):
		return self.getWaitTimeMinutes() + self.getHelpTimeMinutes()

	def __str__(self):
		return "LaIRRequest (" + str(self.requestDateTime) + ") by " + str(self.studentId) + " in 106" + self.courseLetter + " " + self.quarter + " " + str(self.year)

'''
CLASS: BaseObject
-----------------
A base class for all other classes created for reading in CS198 CSV files.  This
class implements base methods that all classes are required to have for CSV parsing.
-----------------
'''
class BaseObject:
	'''
	CLASS METHOD: parseCSV
	----------------------
	Parameters:
		cls - the Class object
		line - an array of attributes read in from CSV

	Returns: an object representing the parsed version of this line, or None
	if the line could not be parsed.
	----------------------
	'''
	@classmethod
	def parseCSV(cls, line):
		return cls(line)

	'''
	METHOD: getKey
	--------------
	Parameters: NA
	Returns: a unique identifier for this object among all other instances of
	this class.
	--------------
	'''
	def getKey(self):
		return self.id

'''
CLASS: Course
-------------
A class that represents a single row in the Courses CSV file.  A Course has the
following instance variables:
	- id: unique identifier
	- course_number: name of course, e.g. "106a"
	- quarter_id: identifier of corresponding Quarter object
-------------
'''
class Course(BaseObject):
	@classmethod
	def parseCSV(cls, line):
		className = line[2]
		# Only return Course objects for A, B and X
		if className == "106a" or className == "106b" or className == "106x":
			return cls(line)
		return None

	def __init__(self, line):
		self.id = int(line[0])
		self.course_number = line[2]
		self.quarter_id = int(line[3])

	def __str__(self):
		return "Quarter #" + str(self.id) + " " + self.course_number + " (" + str(self.quarter_id) + ")"

'''
CLASS: HelpRequest
------------------
A class that represents a single row in the Help Request CSV file.  A HelpRequest
has the following instance variables:
	- id: unique identifier
	- student_relation_id: identifier of corresponding StudentRelation object
	- problem_description: string description of help request issue
	- request_time: date/time request was made
------------------
'''
class HelpRequest(BaseObject):
	@classmethod
	def parseCSV(cls, line):
		requestDateTime = dateutil.parser.parse(line[4][:line[4].index('.')])
		requestQueue = line[5]

		# Only return a new HelpRequest if it is from the Standard (not CLaIR or CS106AJ) Queue.
		if requestQueue == '' or requestQueue == 'Standard':
			return cls(line)
		return None

	def __init__(self, line):
		self.id = int(line[0])
		self.student_relation_id = int(line[1])
		self.problem_description = line[2]
		self.request_time = dateutil.parser.parse(line[4])

	def __str__(self):
		return "HelpRequest #" + str(self.id) + " @ " + str(self.request_time)

'''
CLASS: HelperAssignment
------------------
A class that represents a single row in the Helper Assignment CSV file.  A
HelperAssignment has the following instance variables:
	- id: unique identifier
	- helper_checkin_id: identifier of corresponding HelperCheckin object
	- help_request_id: identifier of corresponding HelpRequest object
	- claim_time: date/time of this request being claimed by this helper
	- close_time: date/time of this request being closed by this helper
------------------
'''
class HelperAssignment(BaseObject):
	def __init__(self, line):
		self.id = int(line[0])
		self.helper_checkin_id = int(line[1])
		self.help_request_id = int(line[2])
		self.claim_time = dateutil.parser.parse(line[3])
		self.close_time = dateutil.parser.parse(line[4])

	# Uniquely identify every HelperAssignment by its help request
	# (Note that some reassigned requests may have multiple HelperAssignments; we remove these)
	def getKey(self):
		return self.help_request_id

	def __str__(self):
		return "HelperAssignment #" + str(self.id) + " claimed @ " + str(self.claim_time) + ", closed @ " + str(self.close_time)

'''
CLASS: HelperCheckin
------------------
A class that represents a single row in the Helper Checkin CSV file.  A HelperCheckin
has the following instance variables:
	- id: unique identifier
	- person_id: unique identifier of this helper
	- check_in_time: date/time this helper checked in
	- check_out_time: date/time this helper checked out
------------------
'''
class HelperCheckin(BaseObject):
	def __init__(self, line):
		self.id = int(line[0])
		self.person_id = int(line[1])
		self.check_in_time = dateutil.parser.parse(line[2])
		self.check_out_time = dateutil.parser.parse(line[3])

	def __str__(self):
		return "HelperCheckin #" + str(self.id) + ", person # " + str(self.person_id) + ", checkin @ " + str(self.check_in_time)

'''
CLASS: Quarter
------------------
A class that represents a single row in the Quarter CSV file.  A Quarter
has the following instance variables:
	- id: unique identifier
	- year: year of this quarter
	- quarter_name: string (e.g. "Fall", "Winter", "Spring")
------------------
'''
class Quarter(BaseObject):
	def __init__(self, line):
		self.id = int(line[0])
		self.year = int(line[1])
		self.quarter_name = line[2]

	def __str__(self):
		return "Quarter: " + str(self.id) + ", " + str(self.year) + ", " + self.quarter_name

'''
CLASS: StaffRelation
------------------
A class that represents a single row in the Staff Relation CSV file.  A StaffRelation
has the following instance variables:
	- id: unique identifier
	- person_id: unique identifier of this helper
	- course_id: unique identifier for the course this helper is staffing for
------------------
'''
class StaffRelation(BaseObject):
	def __init__(self, line):
		self.id = int(line[0])
		self.person_id = int(line[1])
		self.course_id = int(line[2])

	# Uniquely identify StaffRelations by the person_id (no duplicates found)
	def getKey(self):
		return self.person_id

	def __str__(self):
		return "StaffRelation #" + str(self.id) + ", person_id = " + str(self.person_id) + ", course_id = " + str(self.course_id)

'''
CLASS: StudentRelation
------------------
A class that represents a single row in the Staff Relation CSV file.  A StudentRelation
has the following instance variables:
	- id: unique identifier
	- person_id: unique identifier of this student
	- course_id: unique identifier for the course this student is in
------------------
'''
class StudentRelation(BaseObject):
	def __init__(self, line):
		self.id = int(line[0])
		self.person_id = int(line[1])
		self.course_id = int(line[2])

	def __str__(self):
		return "StudentRelation #" + str(self.id) + ", person_id = " + str(self.person_id) + ", course_id = " + str(self.course_id)

'''
FUNCTION: run
-------------
Main script function.  Parses all provided CSV files, creates LaIRRequest objects
from the parsed data, and outputs these LaIRRequests to disk.
-------------
'''
def run():
	courses = readLaIRCSV(COURSES_FILENAME, Course)
	help_requests = readLaIRCSV(HELP_REQUESTS_FILENAME, HelpRequest)
	helper_assignments = readLaIRCSV(HELPER_ASSIGNMENTS_FILENAME, HelperAssignment)
	helper_checkins = readLaIRCSV(HELPER_CHECKINS_FILENAME, HelperCheckin)
	quarters = readLaIRCSV(QUARTERS_FILENAME, Quarter)
	staff_relations = readLaIRCSV(STAFF_RELATIONS_FILENAME, StaffRelation)
	student_relations = readLaIRCSV(STUDENT_RELATIONS_FILENAME, StudentRelation)
	lair_requests = LaIRRequest.createLaIRRequests(courses, help_requests, helper_assignments, helper_checkins, quarters, staff_relations, student_relations)

	print("Writing LaIR Requests to disk....")
	LaIRRequest.saveToFile("dataset", lair_requests, partition=True)
	print("Done.")

'''
FUNCTION: readLaIRCSV
---------------------
Parameters:
	filename - the CSV filename to read in
	Object - the Object to create instances of while parsing this CSV file.  This
	Object must extend BaseObject so that it implements the required parseCSV class
	method and getKey instance method.

Returns: a map from each Object's unique identifier (from getKey()) to that Object,
created from the file with the given name.
---------------------
'''
def readLaIRCSV(filename, Object):
	with open(filename, 'r') as csvfile:
		print("Parsing " + filename + "....")
		csvreader = csv.reader(csvfile)
		next(csvreader, None) # ignore CSV header

		dataMap = {}
		seenIds = set()
		for row in tqdm([row for row in csvreader]):
			parsedRow = Object.parseCSV(row)
			if parsedRow is not None:
				if parsedRow.getKey() not in seenIds:
					dataMap[parsedRow.getKey()] = parsedRow
				elif parsedRow.getKey() in dataMap:
					del dataMap[parsedRow.getKey()]
				seenIds.add(parsedRow.getKey())

		return dataMap


if __name__ == "__main__":
	run()