import random


length = 10000

flightdata = True
trainingdata = True

datatype = 'airline'

trainfile = 'flightdataPrice.txt'
testfile = 'testdataAir.txt'


# Types: pricelow, airline

airlines = ['southwest', 'american', 'united']


def ranEvent():
    time = random.randint(0, 2400)
    date = random.randint(0, 31)
    return [time, date]


def ranFlight():
    time = random.randint(0, 2400)
    date = random.randint(0, 31)
    price = random.randint(0, 500)
    airline = random.randint(0, 2)
    return [time, date, price, airline]


def ranSample():
    f1 = ranFlight()
    f2 = ranFlight()
    f3 = ranFlight()
    e1 = ranEvent()
    return [[e1, f1, f2, f3], e1 + f1 + f2 + f3]


def scaleval(s1):

    s = s1[:]

    s[0][0] = s[0][0] / 2400
    s[0][1] = s[0][1] / 31

    for i in range(1, 4):
        s[i][0] = s[i][0] / 2400
        s[i][1] = s[i][1] / 31
        s[i][2] = s[i][2] / 500
        s[i][3] = s[i][3] / 2

    output = []

    for i in s:
        for v in i:
            output.append(v)

    return output


def findOutput(fsample, tp, airline):
    output = [0, 0, 0]
    sample = fsample[0]
    if tp == 'pricelow':
        highest = [1000, 0]
        for i in sample[1:]:
            x = i[2]
            if x < highest[0]:
                highest[0] = x
                highest[1] = sample[1:].index(i)
        output[highest[1]] = 1
        return output

    if tp == 'airline':
        aln = airlines.index(airline)
        for i in sample[1:]:
            if i[3] == aln:
                output[sample[1:].index(i)] = 1
        return output


def ranSet(tp, airline="southwest"):
    sample = ranSample()
    nsample = scaleval(sample[0])
    output = findOutput(sample, tp, airline)
    return [nsample, output]


def fullDataSet(f):
    tofile = []
    for _ in range(length):
        tofile.append(ranSet(datatype))
    with open(f, 'r+') as fl:
        fl.write(str(tofile))

if flightdata:
    fullDataSet(trainfile)
if trainingdata:
    fullDataSet(testfile)
