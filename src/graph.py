from data.DataLoader import DataLoader
import matplotlib.pyplot as plt
from collections import Counter
from util import make_buckets, make_bucket_mapper

if __name__ == "__main__":
	d = DataLoader()
	d.loadData('../dataset/dataset.npy')

	help_vals = [r.getHelpTimeMinutes() for r in d.laIRRequests]
	wait_vals = [r.getWaitTimeMinutes() for r in d.laIRRequests]

	bucket_vals = [i for i in range(0, 120, 10)] + [float('inf')]

	plt.hist([help_vals, wait_vals], bucket_vals, label=["Help Time", "Wait Time"])
	plt.title("CS106 LaIR Wait and Help Times")
	plt.xlabel("Time (minutes)")
	plt.ylabel("# Requests")
	plt.legend()
	plt.show()