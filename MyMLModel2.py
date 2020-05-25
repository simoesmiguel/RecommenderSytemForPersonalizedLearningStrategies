import csv
from functools import reduce

import jellyfish
import nltk

from Levenshtein import _levenshtein
import matplotlib.pyplot as plt
from joblib.numpy_pickle_utils import xrange
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from pyjarowinkler import distance


#files_common_path = '/Users/miguelsimoes/Documents/Universidade/Tese/Final Data Warehouse/'  # MACOS
files_common_path = 'D:/ChromeDownloads/TeseFolder/Tese/Final Data Warehouse/'      # Windows


def readCSVfile(csv_file_name, d):
    l = []
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=d)
        for row in csv_reader:
            l.append(row)
        return l

def orderdicbyValue(dic, tag): # orders dict in reverse order

    if tag == "levenshtein" or tag == "jaccard" or tag == "hamming" or tag == "soundex" or tag == "damerau":
        return {k: v for k, v in reversed(sorted(dic.items(), key=lambda item: item[1]))}
    elif tag == "jaro" or tag == "lcs":
        return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}

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


def buildInputandOutput(file):
    dimensions = file[0]
    indexes_input_dimensions=[dimensions.index(el) for el in dimensions if el == "cluster" or "Student" in el]
    indexes_output_dimenisons=[dimensions.index(el) for el in dimensions if "Next" in el]

    input, output=[], []

    for line in file[1:]:
        dic_input={}
        dic_output={}
        for index in indexes_input_dimensions:
            dic_input[dimensions[index]] = line[index]
        for index in indexes_output_dimenisons:
            dic_output[dimensions[index]] = line[index]

        input.append(dic_input)
        output.append(dic_output)

        '''
        example of dic_input:
            {'cluster': '2', 'Student skills': '[]', 'Student badges': "['00400211onechallenge', '00800611postonetypeofcreativeresult', ...]", 'Student bonus': "['20308222AttendedQuiz', ...]", 'Student quizzes': '[]', 'Student posts': '[]'}

        example of dic_output : 
            {'Recommended Skills': '[]', 'Recommended badges': "['03502011betherefor50%oflabs', '03000611postfive(ormore)typesofcreativeresults', '01901111makeonepost']"}
        '''

    return input, output


def parseLists(lista_string):
    l = lista_string.replace(" ","").split(",")
    if len(l) > 1:
        l[0] = l[0][1:]
        l[-1] = l[-1][0:len(l[-1])-1]
    else:
        if l[0] != '[]':
            l[0] = l[0][1:len(l[0])-1]
        else: return []

    for i in range(len(l)):
        l[i] = l[i][1:len(l[i])-1]

    return l



def findNeighbors(input, tg_student_info, k_neighbors, tag):
    neighbors = []

    target_student_indicators = [1, 1, 1, 1, 1]  # skills, badges, bonus, quizzes, posts

    target_student_skills = parseLists(tg_student_info.get("Student skills"))
    target_student_badges = parseLists(tg_student_info.get("Student badges"))
    target_student_bonus = parseLists(tg_student_info.get("Student bonus"))
    target_student_quizzes = parseLists(tg_student_info.get("Student quizzes"))
    target_student_posts = parseLists(tg_student_info.get("Student posts"))

    if len(target_student_skills) == 0:
        target_student_indicators[0] = 0
    if len(target_student_badges) == 0:
        target_student_indicators[1] = 0
    if len(target_student_bonus) == 0:
        target_student_indicators[2] = 0
    if len(target_student_quizzes) == 0:
        target_student_indicators[3] = 0
    if len(target_student_posts) == 0:
        target_student_indicators[4] = 0


    for dic in input:
        if dic.get("cluster") == tg_student_info.get("cluster"): # if they belong to the same level
            neighbor_indicators = [0, 0, 0, 0, 0]
            avg_distance_skills, avg_distance_badges, avg_distance_bonus, avg_distance_quizzes, avg_distance_posts = -1, -1, -1, -1, -1


            neighbor_skills = parseLists(dic.get("Student skills"))
            neighbor_badges = parseLists(dic.get("Student badges"))
            neighbor_bonus = parseLists(dic.get("Student bonus"))
            neighbor_quizzes = parseLists(dic.get("Student quizzes"))
            neighbor_posts = parseLists(dic.get("Student posts"))

            if len(neighbor_skills) != 0:
                neighbor_indicators[0] = 1
            if len(neighbor_badges) != 0:
                neighbor_indicators[1] = 1
            if len(neighbor_bonus) != 0:
                neighbor_indicators[2] = 1
            if len(neighbor_quizzes) != 0:
                neighbor_indicators[3] = 1
            if len(neighbor_posts) != 0:
                neighbor_indicators[4] = 1



            new_l = orderElements(target_student_skills, neighbor_skills, tag)
            ##neighbor skills
            if new_l != []:
                avg_distance_skills = calculateAvgDistance(new_l, tag)

            ##neighbor badges
            new_l = orderElements(target_student_badges, neighbor_badges, tag)
            if new_l != []:
                avg_distance_badges = calculateAvgDistance(new_l, tag)


            ##neighbor bonus
            new_l = orderElements(target_student_bonus, neighbor_bonus, tag)
            if new_l != [] :
                avg_distance_bonus = calculateAvgDistance(new_l, tag)


            ##neighbor quizzes
            new_l = orderElements(target_student_quizzes, neighbor_quizzes, tag)
            if new_l != []:
                avg_distance_quizzes = calculateAvgDistance(new_l, tag)

            ##neighbor posts
            new_l = orderElements(target_student_posts, neighbor_posts, tag)
            if new_l != [] :
                avg_distance_posts = calculateAvgDistance(new_l, tag)


            count1, count2 = 0, 0
            for i in range(len(target_student_indicators)):
                if target_student_indicators[i] == 1:
                    count1 += 1
                if neighbor_indicators[i] == 1:
                    count2 += 1

            if count2 >= count1:  #the neighbor has more completed activities than the target student
                all_distances = [avg_distance_skills, avg_distance_badges, avg_distance_bonus, avg_distance_quizzes,
                                 avg_distance_posts]
                total_distance = 0
                for el in all_distances:
                    if el != -1:
                        total_distance += el


                if tag == "levenshtein" or tag == "jaccard" or tag == "hamming" or tag == "soundex" or tag == "damerau":
                    if len(neighbors) == k_neighbors: # if the max number of neighbors was already reached
                        largest_dist = reduce(lambda a, b: a if a > b else b, [el[0] for el in neighbors])

                        if total_distance < largest_dist:
                            el_to_delete = [el for el in neighbors if el[0] == largest_dist][0]
                            neighbors.remove(el_to_delete)
                            neighbors.append((total_distance, dic))
                    else:
                        neighbors.append((total_distance, dic))


                else: # jaro and lcs measures
                    if len(neighbors) == k_neighbors:
                        lowest_dist = reduce(lambda a, b: a if a < b else b, [el[0] for el in neighbors])

                        if total_distance > lowest_dist:
                            el_to_delete = [el for el in neighbors if el[0] == lowest_dist][0]
                            neighbors.remove(el_to_delete)
                            neighbors.append((total_distance, dic))
                    else:
                        neighbors.append((total_distance, dic))

    return neighbors


def makeRecommendations (neighbors_list, train_input, train_output, test_output, tag):
    # neighbors_list = [(0.6666666666666666, {"cluser":2, "Student Skills": [...], "student badges": [...] }), ... ]
    # train_input, train_output : list of dictionaries
    # test_output : dictionary

    #order neighbors list
    if tag == "levenshtein" or tag == "jaccard" or tag == "hamming" or tag == "soundex" or tag == "damerau":
        ordered_n_list = sorted(neighbors_list, key=lambda item: item[0])
    else:
        ordered_n_list = reversed(sorted(neighbors_list, key=lambda item: item[0]))

    indexes = [train_input.index(tupl[1]) for tupl in ordered_n_list]

    recommendations = [train_output[i] for i in indexes] # list of dictionaires


    total_precision, total_recall, total_f1 = [], [], []

    for key in test_output:

        a = [parseLists(dic.get(key)) for dic in recommendations]  # list of lists
        b = list(dict.fromkeys([activity for lista in a for activity in lista])) # list with all the activities without duplicates

        new_l = orderElements(parseLists(test_output.get(key)), b, tag)
        y_test = [el[0] for el in new_l]
        y_pred = [el[1] for el in new_l]

        if y_test != [] and y_pred != []:
            precision = precision_score(y_test, y_pred,
                                        average="micro")  # the less false positives a classifier gives, the higher is its precision.

            recall = recall_score(y_test, y_pred, average="micro")

            #f1 = f1_score(y_test, y_pred, average="micro")
            total_precision.append(precision)
            total_recall.append(recall)
            #total_f1.append(f1)

        else:
            #return 0, 0, 0, recommendations, y_pred, y_test
            total_precision.append(0)
            total_recall.append(0)

    avg_precision = reduce(lambda a, b: a + b, total_precision) / len(total_precision)
    avg_recall = reduce(lambda a, b: a + b, total_recall) / len(total_recall)
    #f1_score = reduce(lambda a, b: a + b, total_f1) / len(total_f1)


    return avg_precision, avg_recall


def scrutinizeData(neighbors, s, ba, q, bo, posts, tag):

    global neighbor_indicators
    global target_student_indicators


    target_student_indicators = [1,1,1,1,1] # skills, badges, bonus, quizzes, posts

    if len(s) == 0:
        target_student_indicators[0] = 0
    if len(ba) == 0:
        target_student_indicators[1] = 0
    if len(bo) == 0:
        target_student_indicators[2] = 0
    if len(q) == 0:
        target_student_indicators[3] = 0
    if len(posts) == 0:
        target_student_indicators[4] = 0


    dic_skills, dic_badges, dic_quizzes, dic_bonus, posts_list = [], [], [], [], []
    lista_all = []

    for neighbor in neighbors: # for all the target student's neighbors

        neighbor_indicators = [0, 0, 0, 0, 0]

        neighbor_skills = parseLists(neighbor.get("Student skills"))
        neighbor_badges =parseLists(neighbor.get("Student badges"))
        neighbor_bonus = parseLists(neighbor.get("Student bonus"))
        neighbor_quizzes = parseLists(neighbor.get("Student quizzes"))
        neighbor_posts = parseLists(neighbor.get("Student posts"))

        if len(neighbor_skills) != 0:
            neighbor_indicators[0] = 1
        if len(neighbor_badges) != 0:
            neighbor_indicators[1] = 1
        if len(neighbor_bonus) != 0:
            neighbor_indicators[2] = 1
        if len(neighbor_quizzes) != 0:
            neighbor_indicators[3] = 1
        if len(neighbor_posts) != 0:
            neighbor_indicators[4] = 1

        avg_distance_skills, avg_distance_badges, avg_distance_bonus, avg_distance_quizzes, avg_distance_posts = -1, -1, -1, -1, -1

        ##neighbor skills
        new_l = orderElements(s, neighbor_skills, tag)
        if new_l != []:
            avg_distance_skills = calculateAvgDistance(new_l, tag)

        ##neighbor badges
        new_l = orderElements(ba, neighbor_badges, tag)
        if new_l != []:
            avg_distance_badges = calculateAvgDistance(new_l, tag)

        ##neighbor bonus
        new_l = orderElements(bo, neighbor_bonus, tag)
        if new_l != []:
            avg_distance_bonus = calculateAvgDistance(new_l, tag)

            ##neighbor quizzes
        new_l = orderElements(q, neighbor_quizzes, tag)
        if new_l != []:
            avg_distance_quizzes = calculateAvgDistance(new_l, tag)

        ##neighbor posts
        new_l = orderElements(posts, neighbor_posts, tag)
        if new_l != []:
            avg_distance_posts = calculateAvgDistance(new_l, tag)

        count1, count2 = 0, 0
        for i in range(len(target_student_indicators)):
            if target_student_indicators[i] == 1:
                count1 += 1
                if neighbor_indicators[i] == 1:
                    count2 += 1

        if count2 == count1: # we found a neighbor of the target student
            all_distances = [avg_distance_skills, avg_distance_badges, avg_distance_bonus, avg_distance_quizzes, avg_distance_posts]
            total_distance=0
            for el in all_distances:
                if el != -1:
                    total_distance+=el

            lista_all.append( ( total_distance, [("skills", neighbor_skills ),("badges", neighbor_badges),("bonus", neighbor_bonus),("quizzes", neighbor_quizzes),("posts",neighbor_posts)]))


    return lista_all

def getNeighbors(cluster, index, file): # get the neighbors of the student which is placed in position "index" from the training file.

        neighbors = []
        if cluster !=3:
            for dic in file:
                if int(dic.get("cluster")) > cluster:
                    neighbors.append(dic)
        else:
            for i in range(len(file)):
                if i != index and int(file[i].get("cluster")) == cluster:
                    neighbors.append(file[i])

        return neighbors


def sortListofTuples(l, tag):
    if tag == "levenshtein" or tag == "jaccard" or tag == "hamming" or tag == "soundex" or tag == "damerau":
        return (sorted(l, key = lambda x: x[0]))
    else:  # jaro and lcs measures
        return reversed(sorted(l, key = lambda x: x[0]))

def recommendSkills(list_skills, target_student_info, kneighbors, n_recommendations):

    # go through all the k nearest neighbors and save the skills that were not yet performed by the target student
    skills_to_recommend, badges_to_recommend, quizzes_to_recommend, bonus_to_recommend, posts_to_recommend = [], [], [], [], []
    neighbors=1

    for tupl in list_skills:
        if neighbors <= kneighbors or (len(skills_to_recommend) == 0 and len(badges_to_recommend)== 0 and len(quizzes_to_recommend)== 0 and len(bonus_to_recommend)== 0 and len(posts_to_recommend)== 0):

            # if the number of neighbors visited is less than the number of neighbors desired to search, or if any recommendation was not found yet
            all_info = tupl[1]

            for tuple in all_info:
                if tuple[0] == "skills":
                    for code in tuple[1]:
                        if code not in target_student_info[0]:
                            skills_to_recommend.append(code)

                elif tuple[0] == "badges":
                    for code in tuple[1]:
                        if code not in target_student_info[1]:
                            badges_to_recommend.append(code)

                elif tuple[0] == "bonus":
                    for code in tuple[1]:
                        if code not in target_student_info[2]:
                            bonus_to_recommend.append(code)

                elif tuple[0] == "quizzes":
                    for code in tuple[1]:
                        if code not in target_student_info[3]:
                            quizzes_to_recommend.append(code)
                else:
                    for code in tuple[1]:
                        if code not in target_student_info[4]:
                            posts_to_recommend.append(code)

            neighbors+=1
        else:
            break


    a = list_occurrences(skills_to_recommend, n_recommendations)
    b = list_occurrences(badges_to_recommend, n_recommendations)
    c = list_occurrences(bonus_to_recommend, n_recommendations)
    d = list_occurrences(quizzes_to_recommend, n_recommendations)
    e = list_occurrences(posts_to_recommend, n_recommendations)

    return [a, b, c, d, e]

def list_occurrences(recommendations, n_skills_to_recommend):
    code_ocurrences = []  # this list saves the number of ocurrences of each element of the list "recommendations"
    for code in recommendations:
        if code not in [el[0] for el in code_ocurrences]:
            code_ocurrences.append((code, recommendations.count(code)))

    final_list = [ tupl for tupl in reversed(sorted(code_ocurrences, key=lambda item: item[1]))] # sort the list by the number of ocurrences

    if len(final_list) <= n_skills_to_recommend:
        return [el[0] for el in final_list]
    else:
        return [el[0] for el in final_list][0:n_skills_to_recommend]


def main():


    l = readCSVfile(files_common_path + "trainindFile_right.csv", ",")
    train_input, train_output = buildInputandOutput(l)

    all_measures =["soundex", "jaccard", "jaro", "damerau"]
    colors_labels = ['b', 'g', 'dimgray', 'red']
    colors_labels2 = ['yellowgreen', 'violet', 'tan', 'maroon']


    for tag in all_measures:
        accuracy_k = [(0,0)]
        print("tag: ",tag)
        #accuracy_k =[]

        for k in range(1,10): # k neighbors
            acertou, errou = 0, 0

            all_recalls = []
            all_precisions = []
            all_f1 = []


            for i in range (len(train_input)):

                student_cluster = int(train_input[i].get("cluster"))
                student_skills = parseLists(train_input[i].get("Student skills"))
                student_badges = parseLists(train_input[i].get("Student badges"))
                student_bonus = parseLists(train_input[i].get("Student bonus"))
                student_quizzes = parseLists(train_input[i].get("Student quizzes"))
                student_posts = parseLists(train_input[i].get("Student posts"))

                n = getNeighbors(student_cluster, i, train_input)

                lista_all = scrutinizeData(n, student_skills, student_badges, student_quizzes, student_bonus, student_posts, tag)

                lista = sortListofTuples(
                    lista_all, tag)  # this list contains all the neighbors of the target student ordered by the distance that each of them is from the target student, as well as all the activities performed by them

                list_all_recommendations = recommendSkills(lista, [student_skills, student_badges,
                                                                                student_bonus, student_quizzes,
                                                                                student_posts] , k, 3)


                #print("l: ",list_all_recommendations)
                #print(train_output[i])
                #print("\n")

                chosen_by_student = parseLists(train_output[i].get("Next Skills"))
                recommended_by_the_system = list_all_recommendations[0]


                '''
                
                new_l = orderElements(chosen_by_student, recommended_by_the_system, tag)
                y_test = [el[0] for el in new_l]
                y_pred = [el[1] for el in new_l]
                #print(y_test)
                #print(y_pred)

                if y_test != [] and y_pred != []:
                    precision = precision_score(y_test, y_pred,
                                                average="micro")  # the less false positives a classifier gives, the higher is its precision.

                    all_precisions.append(precision)
                    #print("precision: ",precision)

                



                '''

                '''
                

                all_recalls.append(r)
                all_precisions.append(precision)
                #all_f1.append(f1)

            mean_recall = reduce(lambda a, b: a + b, all_recalls) / len(all_recalls)
            
            mean_precision = reduce(lambda a, b: a + b, all_precisions) / len(all_precisions)
            #mean_f1 = reduce(lambda a, b: a + b, all_f1) / len(all_f1)


            accuracy_k.append((mean_precision, k))

        '''
                if chosen_by_student != [] and recommended_by_the_system != []:
                    a = chosen_by_student[0]
                    b = recommended_by_the_system[0]

                    if a == b:
                        acertou += 1
                    else:
                        errou += 1


            print("for k = ",k)
            print("acertoou: ", acertou)
            print("errou : ", errou)

        '''

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
    
    '''
