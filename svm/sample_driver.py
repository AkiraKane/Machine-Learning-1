import argparse
import numpy as np 

from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC
class ThreesAndEights:
    """
    Class to store MNIST data, 3 and 8
    """

    def __init__(self, location):
		
		import cPickle, gzip

        # Load the dataset
		f = gzip.open(location, 'rb')

		train_set, valid_set, test_set = cPickle.load(f)

		self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0],:]
		self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0]]

		shuff = np.arange(self.x_train.shape[0])
		np.random.shuffle(shuff)
		self.x_train = self.x_train[shuff,:]
		self.y_train = self.y_train[shuff]

		self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0],:]
		self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0]]
		
		self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0],:]
		self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0]]

		f.close()

def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
	    plt.savefig(outname)
	else:
	    plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='SVM classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	args = parser.parse_args()

	data = ThreesAndEights("../data/mnist.pkl.gz")
	classifier=SVC(C=5,kernel='linear')
	classifier.fit(data.x_train, data.y_train)
	# find the support vectors
	sv = classifier.support_

	y_pred=classifier.predict(data.x_test)
	y_train_score=classifier.score(data.x_train, data.y_train)
	y_test_score=classifier.score(data.x_test, data.y_test)
			
	# -----------------------------------
	# Plotting Examples 
	# -----------------------------------
	indices_3=[]
	indices_8=[]
	for i in sv:
		if data.y_train[i] == 3:
			indices_3.append(i)
		elif data.y_train[i] == 8:
			indices_8.append(i)
	# # Plot image to file 
	mnist_digit_show(data.x_train[ indices_3[15],:], "mnist3.png")
	mnist_digit_show(data.x_train[ indices_8[13],:], "mnist8.png")









