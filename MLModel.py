import csv
from functools import reduce
import numpy as np

from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier


from sklearn import metrics


files_common_path = '/Users/miguelsimoes/Documents/Tese/Final Data Warehouse/'

dic_y_trainEncoded = {}
dic_x_trainEncoded = {}

dic_y_testEncoded = {}
dic_x_testEncoded = {}

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
        index=0
        lista=[]
        for i in range(len(line)):  # build train_input

            if line[i] == "values: ":
                index = i
                train_input.append(lista)
                lista=[]
                break
            else:
                lista.append(line[i])
        '''
        for i in range(index+1, len(line)):  # build train_output
            lista.append(line[i])
            
        train_output.append(lista)
        '''
        train_output.append(line[index+1])

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

def KNNclassifier(xtrain, ytrain, xtest):
    model = KNeighborsClassifier(n_neighbors=1)

    model.fit(xtrain, ytrain)

    y_pred = model.predict(xtest)

    return y_pred

'''
def LinearRegression(xtrain, ytrain, xtest):
    predictor = LinearRegression()
    predictor.fit(X=xtrain, y=ytrain)
    outcome = predictor.predict(X=xtest)

    return outcome
'''
def RandomForest(X_train, y_train, X_test, y_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.50,max_depth = 15, max_leaf_nodes=5).fit(X_train, y_train)

    score= clf.score(X_test, y_test)

    return score


#matches the encoded elements with the uncoded ones
def match(l1, l2, tag, tag2):

    if tag2== "train":
        for i in range(len(l1)):
            if tag == "input":
                dic_x_trainEncoded[str(l1[i])] = l2[i]
            else:
                dic_y_trainEncoded[str(l1[i])] = l2[i]
    else:
        for i in range(len(l1)):
            if tag == "input":
                dic_x_testEncoded[str(l1[i])] = l2[i]
            else:
                dic_y_testEncoded[str(l1[i])] = l2[i]

def trainModel():
    l = readCSVfile(files_common_path + "Student Evaluation/trainingFileSkills.csv", ",")
    new_l = [line[1:] for line in l]
    tinput, toutput = buildTrainLists(new_l)

    #print(tinput)
    l2 = readCSVfile(files_common_path + "Student Evaluation/testFileSkills.csv", ",")
    new_l2 = [line[1:] for line in l2]
    testinput, testoutput = buildTrainLists(new_l2)

    new_tinput, new_testinput = preprocessingData(tinput, testinput)

    #print(new_tinput)

    # encode x_train, y_train, x_test and y_test
    x_train_decoded = [el for line in new_tinput for el in line]
    #y_train_decoded = [el for line in new_toutput for el in line]
    x_train_index = len(x_train_decoded)
    #y_train_index = len(y_train_decoded) + x_train_index
    y_train_index = len(toutput) + x_train_index

    x_test_decoded = [el for line in new_testinput for el in line]
    #y_test_decoded =  [el for line in new_testoutput for el in line]
    x_test_index = len(x_test_decoded) + y_train_index

    all_to_encode = x_train_decoded + toutput + x_test_decoded + testoutput

    clusters = [line[0] for line in l]
    clusters2 = [line[0] for line in l2]

    all_encoded = labelEncoder(all_to_encode)

    x_train_encoded = all_encoded[0 : x_train_index]  # the x_train set ends at x_train_index -1
    y_train_encoded = all_encoded[x_train_index : y_train_index]# the y_train set begins at x_train_index ends at y_train_index -1
    x_test_encoded = all_encoded[y_train_index : x_test_index]
    y_test_encoded = all_encoded[x_test_index : len(all_encoded)]


    ###############################################

    X_train = originalFormat(new_tinput, x_train_encoded)

    #Y_train = originalFormat(new_toutput, y_train_encoded)
    Y_train = y_train_encoded

    X_test = originalFormat(new_testinput, x_test_encoded)

    #Y_test = originalFormat(new_testoutput, y_test_encoded)
    Y_test= y_test_encoded

    for i in range(len(X_train)):
        X_train[i] = [int(clusters[i])] + X_train[i]

    for i in range(len(X_test)):
        X_test[i] = [int(clusters2[i])] + X_test[i]



    #match(X_train, new_tinput, "input", "train")
    #match(Y_train, new_toutput, "output", "train")
    #match(X_test, new_testinput, "input", "test")
    #match(Y_test, new_testoutput, "output", "test")

    '''
    y_pred = KNNclassifier(X_train, Y_train, X_test)
    
    print("labels in y_true that don't appear in y_pred: ", set(Y_test) - set(y_pred))

    # results = confusion_matrix(Y_test, y_pred)

    # print('Confusion Matrix :')
    # print(results)

    print('Accuracy Score :', accuracy_score(Y_test, y_pred))
    print('Report : ')
    print(classification_report(Y_test, y_pred))
    '''


    y_pred = RandomForest(X_train, Y_train, X_test, Y_test)
    print("Score: ", y_pred)

    ######## Just to order the things #########
    '''
    example:
            from this: [15, 10, 14]  :  [ 7 11 15]
            to this  : [15, 10, 14]  :  [15  7 11]

    '''
    '''
    for i in range(len(Y_test)):
        line_test = Y_test[i]
        line_pred = [el for el in y_pred[i]]
        new_test=[]
        already_added=[]

        for j in range(len(line_test)):
            if line_test[j] in line_pred:
                new_test.append(line_test[j])
                already_added.append(j)
        new_test = sorted(new_test)

        for j in range(len(line_test)):
            if j not in already_added:
                new_test.append(line_test[j])
                already_added.append(j)

        Y_test[i] = new_test

    for i in range(len(y_pred)):
        line_test = Y_test[i]
        line_pred = [el for el in y_pred[i]]
        new_pred = []
        already_added=[]

        for j in range(len(line_pred)):
            if line_pred[j] in line_test:
                new_pred.append(line_pred[j] )
                already_added.append(j)
        new_pred = sorted(new_pred)

        for j in range(len(line_pred)):
            if j not in already_added:
                new_pred.append(line_pred[j])
                already_added.append(j)

        y_pred[i] = new_pred
    ##############################################
    
    

    l_final_pred=[]
    for i in range(0,3):
        for el in y_pred:
            l_final_pred.append(el[i])

    l_final_test=[]
    for i in range(0,3):
        for el in Y_test:
            l_final_test.append(el[i])
    '''

    #a = [el for line in y_pred for el in line]
    #b = [el for line in Y_test for el in line]






def main():
    y_pred= trainModel()
    #testModel(y_pred)
















