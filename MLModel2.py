import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import preprocessing


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets
from functools import reduce
from collections import Counter




from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve

files_common_path = '/Users/miguelsimoes/Documents/Tese/Final Data Warehouse/'

def preprocessingData(train, test):

    #for x
    fill_code = "0000000000"
    l = [len(line) for line in train]
    largestlinetrain = reduce(lambda x, y: x if (x > y) else y, l)

    l1 = [len(line) for line in test]
    largestlinetest = reduce(lambda x, y: x if (x > y) else y, l1)

    largestline = reduce(lambda x, y: x if (x > y) else y, [largestlinetrain] + [largestlinetest])

    for line in train:
        while len(line) < largestline:
            line.append(fill_code)

    for line in test:
        while len(line) < largestline:
            line.append(fill_code)

    return train, test

def readCSVfile(csv_file_name, d):
    l = []
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=d)
        line_count = 0
        for row in csv_reader:
            if line_count == 0: line_count += 1
            else:
                l.append(row)
                line_count += 1
        return l


def buildTrainLists(l):
    train_input = []
    train_output = []
    for line in l:
        index = 0
        lista = []
        for i in range(len(line)):  # build train_input

            if line[i] == "values: ":
                index = i
                train_input.append(lista)
                lista = []
                break
            else:
                lista.append(line[i])
        '''
        for i in range(index+1, len(line)):  # build train_output
            lista.append(line[i])

        train_output.append(lista)
        '''
        train_output.append(line[index + 1])

    return train_input, train_output

def labelEncoder(l):
    le = preprocessing.LabelEncoder()
    l_encoded=le.fit_transform(l)
    return l_encoded

def originalFormat(tinput, l_input_encoded):
    index = 0
    new_input = []
    for line in tinput:
        l = []
        for el in line:
            l.append(l_input_encoded[index])
            index += 1
        new_input.append(l)

    return new_input


def logisticRegression(X, y, Xtest, ytest):
    C = 10
    kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

    # Create different classifiers.
    classifiers = {
        'L1 logistic': LogisticRegression(C=C, penalty='l1',
                                          solver='saga',
                                          multi_class='multinomial',
                                          max_iter=10000),
        'L2 logistic (Multinomial)': LogisticRegression(C=C, penalty='l2',
                                                        solver='saga',
                                                        multi_class='multinomial',
                                                        max_iter=10000),
        'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2',
                                                solver='saga',
                                                multi_class='ovr',
                                                max_iter=10000),
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                          random_state=0)
        #,'GPC': GaussianProcessClassifier(kernel)
    }

    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 * 2, n_classifiers * 2))
    plt.subplots_adjust(bottom=.2, top=.95)

    xx = np.linspace(3, 9, 100)
    yy = np.linspace(1, 5, 100).T
    xx, yy = np.meshgrid(xx, yy)

    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X, y)

        y_pred = classifier.predict(Xtest)
        accuracy = accuracy_score(ytest, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))



def SupportVectorMachines(X, y, X_test, y_test):
    svc = SVC()
    svc.fit(X, y)
    svc_disp = plot_roc_curve(svc, X_test, y_test)

    plt.show()


def main():

    l = readCSVfile(files_common_path + "Student Evaluation/trainingFileSkills.csv", ",")
    new_l = [line[1:] for line in l]
    trainInput, trainOutput = buildTrainLists(new_l)

    dic=Counter(trainOutput)
    print(dic)

    #Upsampling
    for key in dic:
        index = trainOutput.index(key)
        count = dic.get(key)
        while count<100:
            trainInput.append(trainInput[index])
            trainOutput.append(key)
            line_l = l[index]
            l.append(line_l)

            count+=1

    dic = Counter(trainOutput)
    print(dic)

    l2 = readCSVfile(files_common_path + "Student Evaluation/testFileSkills.csv", ",")
    new_l2 = [line[1:] for line in l2]
    testInput, testOutput = buildTrainLists(new_l2)

    new_tinput, new_testinput = preprocessingData(trainInput, testInput)


    # encode x_train, y_train, x_test and y_test
    x_train_decoded = [el for line in new_tinput for el in line]
    x_train_index = len(x_train_decoded)
    y_train_index = len(trainOutput) + x_train_index

    x_test_decoded = [el for line in new_testinput for el in line]
    x_test_index = len(x_test_decoded) + y_train_index

    all_to_encode = x_train_decoded + trainOutput + x_test_decoded + testOutput

    clusters = [line[0] for line in l]
    clusters2 = [line[0] for line in l2]

    all_encoded = labelEncoder(all_to_encode)

    x_train_encoded = all_encoded[0: x_train_index]  # the x_train set ends at x_train_index -1
    y_train_encoded = all_encoded[
                      x_train_index: y_train_index]  # the y_train set begins at x_train_index ends at y_train_index -1
    x_test_encoded = all_encoded[y_train_index: x_test_index]
    y_test_encoded = all_encoded[x_test_index: len(all_encoded)]

    ###############################################

    X_train = originalFormat(new_tinput, x_train_encoded)

    Y_train = y_train_encoded

    X_test = originalFormat(new_testinput, x_test_encoded)

    Y_test = y_test_encoded

    for i in range(len(X_train)):
        X_train[i] = [int(clusters[i])] + X_train[i]

    for i in range(len(X_test)):
        X_test[i] = [int(clusters2[i])] + X_test[i]

    X = X_train
    y = Y_train

    Xtest = X_test
    ytest = Y_test

    #logisticRegression(X, y, Xtest, ytest)

    SupportVectorMachines(X, y, Xtest, ytest)
