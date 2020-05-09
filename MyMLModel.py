#encoding:utf8

import csv
from functools import reduce

import jellyfish
import nltk

from Levenshtein import _levenshtein
import matplotlib.pyplot as plt
from joblib.numpy_pickle_utils import xrange
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from pyjarowinkler import distance
import fuzzy

'''
para cada estudante que está presente no test set:
    encontrar quais as entradas do train set que se parecem mais com as atividades dele.
    encontrar as k mais parecidas.
    Recomendar com base nessas.
    Verificar se aquilo que foi recomendado é realmente aquilo que deveria ser recomendado.

'''
files_common_path = '/Users/miguelsimoes/Documents/Universidade/Tese/Final Data Warehouse/'

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


def buildInputandOutput(l):
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

        for i in range(index+1, len(line)):  # build train_output
            lista.append(line[i])

        train_output.append(lista)

        #train_output.append(line[index + 1])

    return train_input, train_output

'''
tg_student_act = target student activities
'''
def findNeighbors(input, tg_student_act, k_neighbors, tag):
    neighbors = []

    if len(tg_student_act)>1:
        for el in input:
            if el[0] == tg_student_act[0] and len(el) > len(tg_student_act): # if they belong to the same level and if the neighbor has more activities completed than the target student

                new_l = orderElements(tg_student_act[1:], el[1:], tag)

                #print("elemento : ", el)
                #print("new_l : ", new_l)

                avg_distance = calculateAvgDistance(new_l, tag)
                #print("avg_distance =  ", avg_distance)

                if tag == "levenshtein" or tag == "jaccard" or tag == "hamming" or tag == "soundex" or tag == "damerau":
                    if len(neighbors) == k_neighbors: # if the max number of neighbors was already reached
                        largest_dist = reduce(lambda a, b: a if a > b else b, [el[0] for el in neighbors])

                        if avg_distance < largest_dist:
                            el_to_delete = [el for el in neighbors if el[0] == largest_dist][0]
                            neighbors.remove(el_to_delete)
                            neighbors.append((avg_distance, el))
                    else:
                        neighbors.append((avg_distance, el))

                else: # jaro and lcs measures
                    if len(neighbors) == k_neighbors:
                        lowest_dist = reduce(lambda a, b: a if a < b else b, [el[0] for el in neighbors])

                        if avg_distance > lowest_dist:
                            el_to_delete = [el for el in neighbors if el[0] == lowest_dist][0]
                            neighbors.remove(el_to_delete)
                            neighbors.append((avg_distance, el))
                    else:
                        neighbors.append((avg_distance, el))

    else:  # if the target student just does not have performed any activities
        for el in input:
            if len(neighbors) < k_neighbors:
                if el[0] == tg_student_act[0] and len(el) > 1:  # if the neighbor and the target student belong to the
                    # same level but the neighbor has performed more activities
                    neighbors.append((0, el))
            else:
                break
    '''
    elif len(neighbors) < k_neighbors:
        for el in input:
            if el[0] == tg_student_act[0] and len(el) == 1:
                neighbors.append((0, el))
    '''

    return neighbors

def calculateAvgDistance(new_l, tag):

    if tag == "levenshtein":
        lst_all_distances = [calculateLevenshteinDistance(tupl[0], tupl[1]) for tupl in new_l]
    elif tag == "jaro":
        lst_all_distances = [calculateJaroDistance(tupl[0], tupl[1]) for tupl in new_l]
    elif tag == "jaccard":
        lst_all_distances = [calculateJaccardDistance(tupl[0], tupl[1]) for tupl in new_l]
    elif tag == "hamming":
        lst_all_distances = [calculateHammingDistance(tupl[0], tupl[1]) for tupl in new_l]
    elif tag == "soundex":
        lst_all_distances = [calculateSoundex(tupl[0], tupl[1]) for tupl in new_l]
    elif tag == "damerau":
        lst_all_distances = [calculateDamerauLevenshteinDistance(tupl[0], tupl[1]) for tupl in new_l]

    else: # tag == lcs
        lst_all_distances = [lcs(tupl[0], tupl[1]) for tupl in new_l]

    # calculate avg distance
    avg_distance = reduce(lambda a, b: a + b, lst_all_distances) / len(lst_all_distances)

    return avg_distance


# Levenshtein Distance
def calculateLevenshteinDistance(s1, s2):
    return _levenshtein.distance(s1, s2)

def calculateJaccardDistance(s1, s2):
    return nltk.jaccard_distance(set(s1), set(s2))

def calculateJaroDistance(s1, s2):
    return distance.get_jaro_distance(s1, s2, winkler=True, scaling=0.1)

def calculateHammingDistance(s1, s2):
    count, i = 0, 0
    if len(s1) > len(s2):
        string1 = s2
        string2 = s1
    else:
        string1 = s1
        string2 = s2

    while i < len(string1):

        if string1[i] != string2[i]:
            count += 1
        i += 1

    count+= abs(len(s1)-len(s2))

    return count

#longest common subsequence similarity
def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in xrange(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

                # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def calculateSoundex(s1,s2):
    string1 = jellyfish.soundex (s1)
    string2 = jellyfish.soundex (s2)

    return calculateHammingDistance(string1, string2)

def calculateDamerauLevenshteinDistance(s1, s2):
    return jellyfish.damerau_levenshtein_distance(s1, s2)

def orderElements(tg_student_act, neighbor_act, tag):
    '''this function joins the skills which were achieved both by target student and its neighbor, but were not
       achieved in the same order. Also, this function joins the most similar skills between the target student and
       its neighbor in order to minimize the avg distance
       For instance:

       student_target_skills = [aa,bb,cd,ef,zx]
       neighbor_skills =       [ce,bb,aa,fg,jr]

       by the end of this code, new_l = [(aa,aa),(bb,bb),(cd,ce),(ef,fg),(zx,jr)]
       Attention! Most of the times, neighbor_skills is a bigger array than student_target_skills
    '''

    copy_act = [el for el in tg_student_act]
    copy_neighbor_act = [element for element in neighbor_act]
    n = []  # tuples list
    elements_to_delete1 = []
    elements_to_delete2 = []

    for code in copy_act:
        for code2 in copy_neighbor_act:
            if code == code2:
                elements_to_delete1.append(code)
                elements_to_delete2.append(code2)
                n.append((code, code2))
                break

    for code1 in elements_to_delete1:
        copy_act.remove(code1)
    for code2 in elements_to_delete2:
        copy_neighbor_act.remove(code2)

    # n2 = list(zip(copy_act, copy_neighbor_act))

    already_added =[]
    n2=[]
    for el in copy_act:
        dic = {}
        for el2 in copy_neighbor_act:
            if el2  not in already_added:

                if tag == "levenshtein":
                    dist = calculateLevenshteinDistance(el, el2)
                elif tag == "jaro":
                    dist = calculateJaroDistance(el, el2)
                elif tag == "jaccard":
                    dist = calculateJaccardDistance(el, el2)
                elif tag == "hamming":
                    dist = calculateHammingDistance(el, el2)
                elif tag == "soundex":
                    dist = calculateSoundex(el, el2)
                elif tag=="damerau":
                    dist = calculateDamerauLevenshteinDistance(el, el2)
                else: #tag == "lcs"
                    dist = lcs(el, el2)

                dic[(el, el2)] = dist

        if dic != {}:
            ordered_dic = orderdicbyValue(dic, tag)

            most_similar_tuple = ordered_dic.popitem()[0]
            already_added.append(most_similar_tuple[1])

            n2.append(most_similar_tuple)

    new_l = n + n2

    return new_l

def orderdicbyValue(dic, tag): # orders dict in reverse order

    if tag == "levenshtein" or tag == "jaccard" or tag == "hamming" or tag == "soundex" or tag == "damerau":
        return {k: v for k, v in reversed(sorted(dic.items(), key=lambda item: item[1]))}
    elif tag == "jaro" or tag == "lcs":
        return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}


def makeRecommendations (neighbors_list, train_input, train_output, test_output, tag):
    # neighbors_list = [(0.6666666666666666, ['1', '10205233ND', '10005133ND', '09504733ND', '10305333ND', '09905033ND']), ... ]

    #order neighbors list
    if tag == "levenshtein" or tag == "jaccard" or tag == "hamming" or tag == "soundex" or tag == "damerau":
        ordered_n_list = sorted(neighbors_list, key=lambda item: item[0])
    else:
        ordered_n_list = reversed(sorted(neighbors_list, key=lambda item: item[0]))


    indexes = [train_input.index(el[1]) for el in ordered_n_list]

    recommendations = list(dict.fromkeys([el for index in indexes for el in train_output[index]]))
    new_l = orderElements(test_output, recommendations, tag)
    y_test = [el[0] for el in new_l]
    y_pred = [el[1] for el in new_l]

    if y_test!=[] and y_pred != []:
        precision = precision_score(y_test, y_pred, average="micro") # the less false positives a classifier gives, the higher is its precision.

        recall = recall_score(y_test, y_pred, average="micro")

        f1 = f1_score(y_test, y_pred,  average="micro")
    else:
        return 0,0,0,recommendations,y_pred,y_test

    return precision, recall, f1, recommendations, y_pred, y_test



def main():

    #l = readCSVfile(files_common_path + "Student Evaluation/trainingFileSkills.csv", ",")
    #l2 = readCSVfile(files_common_path + "Student Evaluation/testFileSkills.csv", ",")

    #l = readCSVfile(files_common_path + "Student Evaluation/trainingFileBadges.csv", ",")
    #l2 = readCSVfile(files_common_path + "Student Evaluation/testFileBadges.csv", ",")

    l = readCSVfile(files_common_path + "Student Evaluation/trainingFileQuizzes.csv", ",")
    l2 = readCSVfile(files_common_path + "Student Evaluation/testFileQuizzes.csv", ",")



    train_input, train_output = buildInputandOutput(l)

    test_input, test_output = buildInputandOutput(l2)


    all_measures =["soundex", "jaccard", "jaro", "damerau"]
    colors_labels = ['b', 'g', 'dimgray', 'red']
    colors_labels2 = ['yellowgreen', 'violet', 'tan', 'maroon']

    for tag in all_measures:
        accuracy_k = [(0,0)]
        print("tag: ",tag)
        #accuracy_k =[]

        for k in range(1,40): # k neighbors

            all_recalls = []
            all_precisions = []
            all_f1 = []


            for i in range (len(test_input)):

                n = findNeighbors(train_input, test_input[i], k, tag)
                r, precision, f1, recommendations, y_pred, y_test = makeRecommendations(n, train_input, train_output, test_output[i], tag)
                all_recalls.append(r)
                all_precisions.append(precision)
                all_f1.append(f1)

            mean_recall = reduce(lambda a, b: a + b, all_recalls) / len(all_recalls)
            mean_precision = reduce(lambda a, b: a + b, all_precisions) / len(all_precisions)
            mean_f1 = reduce(lambda a, b: a + b, all_f1) / len(all_f1)


            accuracy_k.append((mean_precision, k))

        if tag == "soundex":
            l = "Soundex Measure"
        elif tag == "jaccard":
            l = "Jaccard Index"
        elif tag == "jaro":
            l = "Jaro-Winkler"
        elif tag == "damerau":
            l = "Damerau Levenshtein"
        else:
            l= tag

        plt.plot([el[1] for el in accuracy_k], [el[0]*100 for el in accuracy_k], colors_labels[all_measures.index(tag)], label = l)
        #plt.plot([el[3] for el in accuracy_k], [el[1]*100 for el in accuracy_k] , colors_labels2[all_measures.index(tag)], label = l)
        #plt.plot([el[3] for el in accuracy_k], [el[2] for el in accuracy_k])


    plt.ylabel("Precision (%)")
    plt.xlabel("k Neighbors")
    plt.legend()
    plt.title ("Precision Vs (K) Neighbors")
    plt.show()


