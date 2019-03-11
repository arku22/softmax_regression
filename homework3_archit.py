#Implementation of softmax regression

import numpy as np

#loading the design matrix/ training images
X = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics/' + 
            'Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
            '/mnist_train_images.npy')
#print(X.shape)

#Loading the training class labels
Y = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics/' + 
            'Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
            '/mnist_train_labels.npy')

#print(Y.shape)

#Shuffling the order of training examples once
X_Y_matrix = np.hstack((X,Y))
np.random.shuffle(X_Y_matrix)
X = X_Y_matrix[:,0:X.shape[1]]
Y = X_Y_matrix[:,X.shape[1]:]

#print(Y)

#Initialising the parameter matrix
W = np.zeros((X.shape[1], Y.shape[1]))

#Initializing additional variables
num_epoch = 27
n = X.shape[0]
#print(X.shape)
n_tilde = 10
lower_limit = 0
upper_limit = n_tilde
num_runs = n/n_tilde
#print(num_runs)
num_runs = int(num_runs)
#print(num_runs)
#Lambda is regularization strength parameter
Lambda = 0.0001
#alpha is learning rate
alpha = 0.03

for i in range(0, num_epoch):
    lower_limit = 0
    upper_limit = n_tilde
    print("epoch", i)
    for j in range(0, num_runs):
        
        temp_X = X[lower_limit:upper_limit,:]
        temp_Y = Y[lower_limit:upper_limit,:]
        #print(temp_Y.shape)
        
        #Calculating vectorized pre-activation scores
        Z = np.dot(temp_X,W)
        
        #Calculating softmax/y_hat/hypothesis
        hypothesis = np.exp(Z) / ( np.sum(np.exp(Z),axis = 1).reshape(Z.shape[0],1))
        #print(hypothesis.shape)
        
        #Now to find the gradient
        temp_X = temp_X.T
        #print(temp_X.shape)
        #print(hypothesis.shape)
        #print(temp_Y.shape)
        term1 = (1/n_tilde) * np.dot(temp_X,(hypothesis - temp_Y))
        term2 = Lambda * W
        gradient = term1 + term2
        #print(np.mean(gradient))
        #Updating weights
        W = W - alpha*gradient
        #print(np.max(W))
    
        #Re-initialise loop limits
        lower_limit = upper_limit
        upper_limit = upper_limit + n_tilde
        

#Now to evaluate the performance on the validation set

#We use the parameter matrix found using SGD to come up with a new hypothesis


#check over validation set
#X_check = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics/' + 
            #'Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
            #'/mnist_validation_images.npy')
            
#Y_check = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics/' + 
            #'Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
            #'/mnist_validation_labels.npy')

#Check over test set            
X_check = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics/' + 
            'Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
            '/mnist_test_images.npy')
            
Y_check = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics/' + 
            'Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
            '/mnist_test_labels.npy')

Z = np.dot(X_check,W)
hypothesis = np.exp(Z) / (np.sum(np.exp(Z),axis = 1).reshape(Z.shape[0],1))

total_examples = hypothesis.shape[0]

prediction = np.argmax(hypothesis,axis =1)
#print(prediction)
#print(prediction.shape) 
actual = np.argmax(Y_check,axis =1)
#print(actual)
#print(actual.shape)
count = np.sum(prediction == actual)
        
        
print("Number of matches = " + str(count))
print('Total number of examples = ' + str(total_examples))
performance = (count/total_examples) * 100
print('Machine performance = ' + str(performance) + ' %')
    

        
        
