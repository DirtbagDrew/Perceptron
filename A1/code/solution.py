import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from helper import *
from decimal import Decimal


'''
Homework1: perceptron classifier
'''
def sign(x):
    return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#

def show_images(data):
    '''
    This function is used for plot image and save it.
    
    Args:
    data: Two images from train data with shape (2, 16, 16). The shape represents total 2
    	      images and each image has size 16 by 16. 
    
    Returns:
    		Do not return any arguments, just save the images you plot for your report.
    '''
    # prints and saves the first image
    print('image 1')
    imgplot=plt.imshow(data[0])
    plt.show()
    imgplot=plt.imshow(data[0])
    plt.savefig('img1.png')
    plt.gcf().clear()
    # prints and saves the second image
    print('image 2')
    imgplot=plt.imshow(data[1])
    plt.show()
    imgplot=plt.imshow(data[1])
    plt.savefig('img2.png')
    plt.gcf().clear()
    
    


def show_features(data, label):
    '''
    This function is used for plot a 2-D scatter plot of the features and save it. 

    Args:
        data: train features with shape (1561, 2). The shape represents total 1561 samples and 
	     each sample has 2 features.
    label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
    Returns:
        Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
	'''
    
    size=len(data)
    for x in range(0,size):
        if label[x] == -1:
            plt.plot(data[x],'b+')
        elif label[x] == 1:
            plt.plot(data[x],'r*')
    print()
    print('features plot')
    plt.show()
    plt.gcf().clear()

def perceptron(data, label, max_iter, learning_rate):
    w = np.zeros((1,max_iter))
    '''
	The perceptron classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (1, 3). You must initilize it with w = np.zeros((1,d))
	'''
    print(str(data[0]))
    
    for i in range (max_iter):
        for j in range(len(data)):
            x, y = (str(data[j]).split())
            x=float(x)
            #y1, y2 = y.split("-")
            #print(y)
            #y=float(y)
            #if(np.dot(x,w)*y) <= 0:
            #    w=w+learning_rate*x*y
    #return w
        


def show_result(data, label, w):
	'''
	This function is used for plot the test data with the separators and save it.
	
	Args:
	data: test features with shape (424, 2). The shape represents total 424 samples and 
	      each sample has 2 features.
	label: test data's label with shape (424,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the image you plot for your report.
	'''


#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
    n, _ = data.shape
    mistakes = 0
    for i in range(n):
        if sign(np.dot(data[i],np.transpose(w))) != label[i]:
            mistakes += 1
            return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
    #get data
    traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
    train_data,train_label = load_features(traindataloc)
    test_data, test_label = load_features(testdataloc)
    #train perceptron
    w = perceptron(train_data, train_label, max_iter, learning_rate)
    train_acc = accuracy_perceptron(train_data, train_label, w)	
    #test perceptron model
    test_acc = accuracy_perceptron(test_data, test_label, w)
    return w, train_acc, test_acc


