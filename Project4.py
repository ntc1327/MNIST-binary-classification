import numpy as np
from sklearn.metrics import classification_report
np.random.seed(27)  # 27 since it's my favorite number


# simple sigmoid function for activation function
def sigmoidFunction(input):
    change = 1 / (1 + np.exp(-input))
    return change


# modified entropy function to calculate loss
def entropy(y, a):
    numExamples = y.shape[1]
    loss = -(1 / numExamples) * (np.sum(np.log(a) * y) + np.sum(np.log(1 - a) * (1 - y)))

    return loss


# number of examples for each dataset
numTrain = 12665
numTest = 2115

# load in datasets
trainData = np.loadtxt('mnist_train_0_1.csv', dtype=str, delimiter=',')
testData = np.loadtxt('mnist_test_0_1.csv', dtype=str, delimiter=',')

# split datasets into the elements (x) and the values (y)
xTrain = np.asarray(trainData[1:, 1:], dtype='float')
xTest = np.asarray(testData[1:, 1:], dtype='float')
yTrain = np.asarray(trainData[1:, 0:1], dtype='float')
yTest = np.asarray(testData[1:, 0:1], dtype='float')

# transpose elements for easier manipulation, reshape values to fit arrays
xTrain = xTrain.T
xTest = xTest.T
yTrain = yTrain.reshape(1, numTrain - 1)
yTest = yTest.reshape(1, numTest - 1)

# switch 0s to be true (1) and 1s to be false (0) so the entropy function has an easier time
alterY = np.zeros(yTrain.shape)
alterY[np.where(yTrain == 0.0)[0]] = 1
yTrain = alterY

# sets learning rate, gets the number of pixels (784) and sets number of hidden nodes
learnRate = 1
numPixels = xTrain.shape[0]
numHiddenNodes = 4

# initialize weights
w1 = np.random.randn(numHiddenNodes, numPixels)
w2 = np.random.randn(1, numHiddenNodes)
b1 = np.zeros((numHiddenNodes, 1))
b2 = np.zeros((1, 1))

for i in range(20):

    # z1 and z2 are basically w^T * x + b
    # a1 and a2 are activation functions, sigmoid in this case
    z1 = np.matmul(w1, xTrain) + b1
    a1 = sigmoidFunction(z1)
    z2 = np.matmul(w2, a1) + b2
    a2 = sigmoidFunction(z2)

    # use entropy to calculate the loss
    cost = entropy(yTrain, a2)

    # begin the weight updates, get errors and derivation of activation function for first layer
    dz2 = a2 - yTrain
    dw2 = (1 / numTrain) * np.matmul(dz2, a1.T)
    db2 = (1 / numTrain) * np.sum(dz2, axis=1, keepdims=True)

    # repeat for second layer, derivation must use sigmoid function since we're a layer in now
    da1 = np.matmul(w2.T, dz2)
    dz1 = da1 * sigmoidFunction(z1) * (1 - sigmoidFunction(z1))
    dw1 = (1 / numTrain) * np.matmul(dz1, xTrain.T)
    db1 = (1 / numTrain) * np.sum(dz1, axis=1, keepdims=True)

    # finally update weights
    # dw1 and dw2 at this point are error * derivative of activation function * X
    # db1 and db2 are 1/number of examples * summation of theta * activation * (1 - activation)
    w1 = w1 - learnRate * dw1
    w2 = w2 - learnRate * dw2
    b1 = b1 - learnRate * db1
    b2 = b2 - learnRate * db2

    # keeps track of cost for each iteration
    print("Iteration", i, "cost: ", cost)


# perform w^T * x + b with the test data
z1 = np.matmul(w1, xTest) + b1
a1 = sigmoidFunction(z1)
z2 = np.matmul(w2, a1) + b2
a2 = sigmoidFunction(z2)

# for each element, determines if it can be accepted as a zero or not
# true if it can be a zero, false if not close enough to accepted threshold
predictions = (a2 > .97)[0, :]

# combs through yTest and if the value is 0, sets it to true, 1 becomes false
actualVals = (yTest == 0)[0, :]

# outputs what was accepted as a 0 (true) and what was rejected (false)
print(classification_report(predictions, actualVals))
