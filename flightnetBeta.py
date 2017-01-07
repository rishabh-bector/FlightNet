from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import numpy as np

airlines = {'southwest': 0, 'american': 1, 'united': 2}

trainfile = 'flightdataPrice.txt'
testfile = 'testdataAir.txt'

epochs = 100

Recall = True
RecallFile = 'AirlineState.h5'

tp = 'airline'


class Network:

    def __init__(self, recall=True):

        if recall:
            self.model = load_model(RecallFile)

        else:
            self.model = Sequential()

            self.model.add(Dense(15, input_dim=14, activation='sigmoid'))
            self.model.add(Dense(30, activation='sigmoid'))
            self.model.add(Dense(10, activation='sigmoid'))
            self.model.add(Dense(3, activation='softmax'))

            self.model.compile(loss='binary_crossentropy',
                               optimizer='nadam')

    def testOnData(self, data, tp):  # Expects inputs + outputs
        total = len(data[0])
        correct = 0
        predictions = self.predict(data[0]).tolist()
        # print(data[0].tolist())
        # print(predictions)

        if tp == 'pricelow':  # Convert to 0s and 1s for analysis (PRICELOW)
            for s in range(len(predictions)):
                biggest = [0, 0]
                for value in predictions[s]:
                    if value > biggest[0]:
                        biggest[0] = value
                        biggest[1] = predictions[s].index(value)

                output = [0, 0, 0]
                output[biggest[1]] = 1
                predictions[s] = output

        if tp == 'airline':
            output = [0, 0, 0]
            # print(predictions)
            for s in range(len(predictions)):  # Convert to 0s and 1s for analysis (AIRLINE)
                for value in predictions[s]:
                    if value >= 0.4:
                        predictions[s][predictions[s].index(value)] = 1
                    else:
                        predictions[s][predictions[s].index(value)] = 0
        # print(predictions)
        # print(data[1].tolist())
        for s in range(len(predictions)):
            if predictions[s] == data[1][s].tolist():
                correct += 1

        return [correct, total]

    def train(self, inputs, outputs):
        self.model.fit(inputs, outputs, nb_epoch=epochs)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def saveModel(self):
        self.model.save(RecallFile)


class Flight:

    def __init__(self, time, day, price, airline):
        self.time = time / 2400
        self.day = day / 31
        self.price = price / 1000
        self.airline = airlines[airline] / 3

    def getData(self):
        return [self.time, self.day, self.price, self.airline]


class Event:

    def __init__(self, time, day):
        self.time = time / 2400
        self.day = day / 31

    def getData(self):
        return [self.time, self.day]


# Training data #

def getData(filen):
    inputs = []
    outputs = []
    fileContents = ''
    with open(filen, 'r+') as fl:
        fileContents = fl.read()
    fileContents = eval(fileContents)
    for dataset in fileContents:
        inputs.append(dataset[0])
        outputs.append(dataset[1])
    return [np.array(inputs), np.array(outputs)]


# Make Model
Model = Network(recall=Recall)
traindata = getData(trainfile)
testdata = getData(testfile)

# Train Model
# Model.train(traindata[0], traindata[1])

# Test Model
print('\nTesting on data :: ' + testfile)
print('Saved Brain     :: ' + RecallFile)
outputs = Model.testOnData(testdata, tp)
print('Samples         :: ' + str(len(testdata[0])))
print('Accuracy        :: ' + str(outputs[0] / outputs[1] * 100) + '%\n')

# Save Model
Model.saveModel()
