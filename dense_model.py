# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# # load the dataset
dataset = loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', delimiter=',')
# # split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
print(X.shape)
print(y)
# # define the keras model
model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))# 12 = Number of units
model.add(Dense(16, activation='relu'))
model.add(Dense(12,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(4,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# # # compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])# Adam optimization is a stochastic gradient descent method that is based
# # # evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
# pip install tensorflow
# pip install keras 2.2.4

# Split the data in training and test sets
# Change the architecture and compare the accuracy performance over 10 experiment runs.
