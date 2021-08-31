import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class ConstantValues:
    def __init__(self, typeIIError, typeIError, batchLimit):
        self.typeIIError = typeIIError
        self.typeIError = typeIError
        self.batchLimit = batchLimit


def setupConstants(typeIIError, typeIError, batchLimit):
    ConstantObject = ConstantValues(typeIIError, typeIError, batchLimit)
    return ConstantObject


# Note: default RNG is new way of getting random numbers by Numpy: "New code should use the binomial method of a default_rng() instance instead; please see the Quick Start." https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html


def setupPopulation(iterations, p):
    """
    Input:
        iterations (int): the size of population
        p (float): the infection rate
    Output:
        populationArray (array): 2d array where the first column is for id and the second column
        is the condition, where 1 stands for infection and 0 stands for uninfection
    """
    idArray = np.arange(iterations)  # idArray is the id of each person
    # randomTable is the random number for each person, 1 stands for infected and 0 stands for uninfected
    randomTable = np.random.default_rng().binomial(n=1, p=p, size=iterations)
    # combines ID array and Infection Array
    populationArray = (np.column_stack((idArray, randomTable)))
    return populationArray


# go through testing of population
def optimalBatchSize(p, Constants, initialGuess=2):
    """
    A function that optimizes the best batch size for one batch test given the infection rate

    Inputs:
        p(float): infection rate
        typeIIError(float): the probability of type II error
        typeIError(float):  the probability of type I error
        batch_limit (int): the upper limit of batch size
        initialGuess(float): the initial guess for batch size
    Output:
        nSolution(float): the optimal batch size
    """
    q = 1 - p  # rate of no infection

    def minimization(n): return n * (q ** (n/2)) - (- (1 -
                                                       Constants.typeIError - Constants.typeIIError) * np.log(q)) ** (-1/2)
    # solves the minimization equation
    tempNSolution = float(fsolve(minimization, initialGuess))
    # solves if we should floor or ceil the solution because we need integer batch size
    floor, ceiling = np.floor(tempNSolution), np.ceil(tempNSolution)
    def function(batchSize): return 1/batchSize + 1 - Constants.typeIIError - \
        (1 - Constants.typeIIError - Constants.typeIError) * (1 - p) ** (batchSize)
    if function(floor) < function(ceiling):
        nSolution = int(floor)
    else:
        nSolution = int(ceiling)
    return min(nSolution, int(Constants.batchLimit))


def sequentialTesting(populationArray, stopRule, p, batchSize, Constants, seqRepeat=1, threshold=1, sequential=True):
    """
    A function gives the test results to a subject array and the total number of
    test-kit consumption and the individual testing number given the subject array,
    the stop rule, the batch size, the probability of type II error, the probability of
    Type I error, and the number of repeatition, the probability threshold, and
    setting of sequence testing or not.

    Input:
        populationArray(Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        stopRule (int): the number of postive batches to enter individual testing
        p (float): infection rate
        batchSize (int): batch size
        Constants (ConstantValues Object):
            typeIIError (float): probability of type II error
            typeIError (float): probability of type I error
            batchLimit (int): the max size the batch can be
        seqRepeat (int): the number of repetitions for sequential testing
        threshold (float): if the infection rate of a batch is beyond threshold, the subjects on that batch will enter individual testing phase
        sequential (boolean): True stands for sequential testing. The test will end when the test result is positive or past the number of repetition. 
        False stands for simutanlous testing with majority voting.
    Output:
        testResults (Numpy Array): an array contains subjects' id and test results
        totalTestConsumption (int): the total test consumption
        individualTestConsumption (int): the test consumption for individual testings

    """
    tempList = []
    negativeInfoList = []
    positiveInfoList = []
    totalTestConsumption = 0  # same as consum
    temp = {
        'populationSubset': populationArray,
        'numNegBatch': 0,
        'numPosBatch': 0,
        'p': p,
        'batch_size': batchSize
    }
    tempList.append(temp)
    newList = []
    negativeBatches = []
    positiveBatches = []
    while len(tempList) > 0:  # len of temp list right now is 1 because we added the temp populationSubset structuref
        for i in tempList:  # which is 1
            negArr, posArr, testConsum, p0, p1, n0, n1 = helperFunction(
                i['populationSubset'], i['p'], i['batch_size'], Constants)
            temp0 = {
                'populationSubset': negArr,
                'numNegBatch': i['numNegBatch'] + 1,
                'numPosBatch': i['numPosBatch'],
                'p': p0,
                'batch_size': n0
            }
            temp1 = {
                'populationSubset': posArr,
                'numNegBatch': i['numNegBatch'],
                'numPosBatch': i['numPosBatch'] + 1,
                'p': p1,
                'batch_size': n1
            }
            if len(temp0['populationSubset']) > 0:
                if temp0['numNegBatch'] >= stopRule:
                    negativeInfoList.append(temp0)
                else:
                    newList.append(temp0)
            if len(temp1['populationSubset']) > 0:
                # threshold is 1 (probability threshold)
                if temp1['numPosBatch'] >= stopRule or temp1['p'] >= threshold:
                    positiveInfoList.append(temp1)
                else:
                    newList.append(temp1)
            totalTestConsumption += testConsum
        tempList = newList
        newList = []
    for i in negativeInfoList:
        negativeBatches.append(i['populationSubset'])
    negativeBatches = np.concatenate(negativeBatches)
    for i in positiveInfoList:
        positiveBatches.append(i['populationSubset'])
    positiveBatches = np.concatenate(positiveBatches)
    # I think you set to 0 because the test will assume everyone here is negative
    negativeBatches[:, 1] = 0
    positiveBatches, individualTestConsumption = conventionalTest(positiveBatches, Constants, seqRepeat)
    totalTestConsumption += individualTestConsumption
    testResults = np.concatenate((positiveBatches,negativeBatches))
    #sorts testResults by ID
    testResults = testResults[testResults[:,0].argsort()]
    testResults = testResults.astype('int64')
    return testResults, totalTestConsumption, individualTestConsumption

def conventionalTest(populationArray, Constants, seqRepeat=1, sequential=True):
    """
    A function gives the test results to a subject array given the probability of
    type II error, the probability of Type I error, and the number of repeatition,
    and setting of sequence testing or not.

    Input:
        populationArray(Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        Constants (ConstantValues Object): 
            typeIIError (float): probability of type II error
            typeIError (float): probability of type I error 
            batchLimit (int): the max size the batch can be
        seqRepeat (int): the number of repetitions
       sequential (boolean): True stands for sequential testing. The test will end when the test result is positive or past the number of repetition. False stands for simutanlous testing with majority voting.

    Output:
        testResults (Numpy Array): a Numpy array containining subjects' id and test results
        totalTestConsum (int) : the total test consumption
    """
    testResults = np.zeros(populationArray.shape, dtype=np.int)
    randomTable = np.random.uniform(0, 1, (populationArray.shape[0],seqRepeat))
    #sequential testing, end testing when the test result is positive or past the number of repetitions
    if sequential == True:
        totalTestConsumption = 0
        for i in range(len(populationArray)):
            testResult = 0
            counter = 0
            individual = populationArray[i,1]
            while counter < seqRepeat and testResult == 0:
                randomNum = randomTable[i,counter]
                totalTestConsumption += 1
                #individual is positive
                if individual == 1:
                    testResult = 1 if randomNum > Constants.typeIIError else 0
                #individual is negative
                else:
                    testResult = 1 if randomNum < Constants.typeIError else 0
                counter += 1
            #individual id    
            testResults[i, 0] = populationArray[i, 0]
            #test result
            testResults[i, 1] = testResult
    #simutaneously testing, test all repetitions at once
    else:
        for i in range(len(populationArray)):
            totalPositives = 0
            for j in range(seqRepeat):
                randomNum = randomTable[i,j]
                if populationArray[i,1] == 1:
                    testResult = 1 if randomNum > Constants.typeIIError else 0
                else:
                    testResult = 1 if randomNum < Constants.typeIError else 0
                totalPositives += testResult
            testResults[i, 0] = populationArray[i, 0]
            if totalPositives >= seqRepeat/2:
                testResults[i, 1] = 1
            else:
                testResults[i, 1] = 0
            totalTestConsumption = len(populationArray) * seqRepeat
    return testResults, totalTestConsumption
def helperFunction(populationArray, p, batchSize, Constants):
    """
    The helperfunction is a handy function to give the list of subjects on the
    negative batch(es), the list of subjects on the postive batch(es), the 
    test-kit consumption, the infection rate on the negative batches, the 
    infection rate on the positive batches, the optimal batch size for
    negative batches and the optimal batch size for positive batches.

    Input: 
        populationArray(Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        p (float): Infection rate
        batchSize (int): batch size
        Constants (ConstantValues Object): 
            typeIIError (float): probability of type II error
            typeIError (float): probability of type I error 
            batchLimit (int): the max size the batch can be
    Output:
        negativeArray (Numpy Array): an array of subjects on the negative batch(es)
        postiveArray (Numpy Array): an array of subjects on the postive batch(es)
        testConsumption (int): the number of test-kit consumptions
        pNegative (float): the infection rate on the negative batches
        pPositive (float): the infection rate on the positive batches
        nNegative (float): the optimal batch size for the negative batches
        nPositive (float): the optimal batch size for the positive batches
    """
    # p is infection rate
    # n is optimal batch size
    batchSize = min(batchSize, Constants.batchLimit)
    pNegative = infectionRateNegativeBatch(p, batchSize, Constants)
    pPositive = infectionRatePositiveBatch(p, batchSize, Constants)
    nNegative = optimalBatchSize(pNegative, Constants)
    nPositive = optimalBatchSize(pPositive, Constants)
    if populationArray == np.array([]):
        return (np.array([]), np.array([]), pNegative, pPositive, nNegative, nPositive)
    negativeArray, positiveArray, testConsumption = batchSplit(
        populationArray, batchSize, Constants)
    return (negativeArray, positiveArray, testConsumption, pNegative, pPositive, nNegative, nPositive)


def batchSplit(populationArray, batchSize, Constants):
    """
    A function that splits the Population into two batches, negative or positive with the test-kit 
    consumption given the probability of type II error and the 
    probability of Type I error.

    Input:
        populationArray(Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        batchSize (int): batch size
        Constants (ConstantValues Object): 
            typeIIError (float): probability of type II error
            typeIError (float): probability of type I error 
            batchLimit (int): the max size the batch can be
    Output:
        negativeBatch (Numpy Array): an array of subjects on the negative batch(es)
        positiveBatch (Numpy Array): an array of subjects on the postive batch(es)
        testConsumption (int): the number of test-kit consumptions

    """
    negativeBatch = []  # list of negative Batches
    positiveBatch = []  # list of positive Batches
    # this is also the number of initial batches
    testConsumption = np.ceil(len(populationArray) / batchSize)
    randomTable = np.random.uniform(size=int(testConsumption))
    for count, tempBatch in enumerate(np.array_split(populationArray, testConsumption)):
        # if the batch is positive
        if 1 in (tempBatch[:, 1]):  # [:,1] is the second column of numpy array
            if randomTable[count] > Constants.typeIIError:
                positiveBatch.append(tempBatch)
            else:
                negativeBatch.append(tempBatch)
        else:  # if the batch is negative
            if randomTable[count] > Constants.typeIError:
                negativeBatch.append(tempBatch)
            else:
                positiveBatch.append(tempBatch)

    negativeBatch = np.concatenate(negativeBatch) if len(
        negativeBatch) > 0 else np.array([])  # concatenates the numpy negative batches
    positiveBatch = np.concatenate(positiveBatch) if len(
        positiveBatch) > 0 else np.array([])  # concatenates the numpy positive batches
    return (negativeBatch, positiveBatch, testConsumption)


def infectionRateNegativeBatch(p, batchSize, Constants):
    """
    Given infection rate, batch size, prob of type II error and prob of type I error, this
    function gives the infection rate on the negative batch.

    Input:
        p (float): infection rate
        batchSize (int): the batch size
        Constants (ConstantValues Object): 
            typeIIError (float): probability of type II error
            typeIError (float): probability of type I error 
            batchLimit (int): the max size the batch can be

    Output:
        (float): the infection rate on the negative batch
    """
    q = 1 - p
    r = Constants.typeIIError * (1 - q ** batchSize)/((1 - Constants.typeIError)
                                                      * q ** batchSize + Constants.typeIIError * (1 - q**batchSize))
    return p*r/(1 - q**batchSize)


def infectionRatePositiveBatch(p, batchSize, Constants):
    """
    Given infection rate, batch size, prob of type II error and prob of type I error, this
    function gives the infection rate on the positive batch.

    Input:
        p (float): infection rate 
        batchSize (int): the batch size
        Constants (ConstantValues Object): 
            typeIIError (float): probability of type II error
            typeIError (float): probability of type I error 
            batchLimit (int): the max size the batch can be

    Output:
        (float): the infection rate on the positive batch
    """
    q = 1 - p
    r = (1 - Constants.typeIIError) * (1 - q ** batchSize)/(Constants.typeIError *
                                                            q ** batchSize + (1 - Constants.typeIIError) * (1 - q**batchSize))
    return p*r/(1 - q**batchSize)

def resultsToDataFrame(populationArrayIterations, **kwargs):
    """
    a helper function provides convenient results for a sequential testing, returns a dataframe with 
    accuracy, sensitivity, specificity, PPV, NPV, totalTestConsumption, individualTestConsumption, and batchTestConsumption

    Input:
        populationArray(Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
    Output:
        result (DataFrame): a dataframe contains important evaluation metrics for the test method 
    """
    iterations= len(populationArrayIterations)
    accuracy = np.zeros(iterations)
    sensitivity = np.zeros(iterations)
    specificity = np.zeros(iterations)
    #positive predictive value
    ppv = np.zeros(iterations)
    #negative predictive value
    npv = np.zeros(iterations)
    totalTestConsumption = np.zeros(iterations)
    individualTestConsumption = np.zeros(iterations)
    batchTestConsumption = np.zeros(iterations)
    for i in range(iterations):
        testResultsArray, totalTestConsumption[i], individualTestConsumption[i] = sequentialTesting(populationArrayIterations[i], **kwargs)
        testing = testResultsArray[:,1]
        reality = populationArrayIterations[i][:,1]
        accuracy[i] = np.mean(reality == testing)
        sensitivity[i] = recall_score(reality, testing)
        specificity[i] = specificity_score(reality, testing)
        ppv[i] = precision_score(reality, testing)
        npv[i] = npv_score(reality, testing)
        batchTestConsumption = totalTestConsumption[i] - individualTestConsumption[i]
        
        result = {'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'totalTestConsumption': totalTestConsumption,
        'individualTestConsumption': individualTestConsumption,
        'batchTestConsumption': batchTestConsumption
        }
    return pd.DataFrame(result)


#statistic metrics (I didn't edit the code because I'ma just assume this is a math equation)
def specificity_score(y_true, y_pred):
    """
    A function provides specificty given the prediction and the truth 
    """
    tn, fp, _, _ = confusion_matrix(y_true = y_true,
    y_pred = y_pred).ravel()
    return tn/(tn + fp)

def npv_score(y_true, y_pred):
    """
    A function provides npv given the prediction and the truth 
    """
    tn, _, fn, _ = confusion_matrix(y_true = y_true,
    y_pred = y_pred).ravel()
    return tn/(tn + fn)