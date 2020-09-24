import csv
from functools import reduce
from statistics import median

import jellyfish
import nltk

from Levenshtein import _levenshtein
import matplotlib.pyplot as plt
from joblib.numpy_pickle_utils import xrange
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from pyjarowinkler import distance
import pandas as pd
import numpy as np
import seaborn as sns


files_common_path = '/Users/miguelsimoes/Documents/Universidade/Tese/Final Data Warehouse/'  # MACOS
#files_common_path = 'D:/ChromeDownloads/TeseFolder/Tese/Final Data Warehouse/'      # Windows


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
            {'cluster': '2', 'Student skills': '[]', 'Student badges': "['00400211onechallenge:250', '00800611postonetypeofcreativeresult:400', ...]", 'Student bonus': "['20308222AttendedQuiz:200', ...]", 'Student quizzes': '[]', 'Student posts': '[]'}

        example of dic_output : 
            {'Next Skills': '[]', 'Next badges': "['03502011betherefor50%oflabs:250', '03000611postfive(ormore)typesofcreativeresults:600', '01901111makeonepost:100']"}
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





def getIndicators(skills, badges, bonus, quizzes, posts):
    indicators = [0, 0, 0, 0, 0]

    if len(skills) != 0:
        indicators[0] = 1
    if len(badges) != 0:
        indicators[1] = 1
    if len(bonus) != 0:
        indicators[2] = 1
    if len(quizzes) != 0:
        indicators[3] = 1
    if len(posts) != 0:
        indicators[4] = 1

    return indicators

def buildActivityXpsDic(all_acts):

    # these dictionaries have the job to keep the info about all the activities and the correspondent collected Xp's
    skills_dic, badges_dic, bonus_dic, quizzes_dic = {}, {}, {}, {}

    index = 0
    for lista in all_acts:
        for act in lista:
            l = act.split(":")
            if l==[]:
                break
            if index == 0:
                skills_dic[l[0]] = l[1]
            elif index == 1:
                badges_dic[l[0]] = l[1]
            elif index == 2:
                bonus_dic[l[0]] = l[1]
            elif index == 3:
                quizzes_dic[l[0]] = l[1]


        index += 1

    return skills_dic, badges_dic,bonus_dic, quizzes_dic

def scrutinizeData(neighbors, s, ba, q, bo, posts, tag):

    global neighbor_indicators
    global target_student_indicators


    # these dictionaries have the job to keep the info about all the activities and the correspondent collected Xp's
    skills_dic, badges_dic, bonus_dic, quizzes_dic = buildActivityXpsDic([s, ba, bo, q])

    target_student_indicators = getIndicators(s, ba, bo, q, posts)


    lista_all = []
    if len(neighbors) ==0:
        print("A LISTA DE VIZINHOS ESTÁ VAZIA")


    for neighbor in neighbors: # for all the target student's neighbors

        neighbor_skills = parseLists(neighbor.get("Student skills"))
        neighbor_badges =parseLists(neighbor.get("Student badges"))
        neighbor_bonus = parseLists(neighbor.get("Student bonus"))
        neighbor_quizzes = parseLists(neighbor.get("Student quizzes"))
        neighbor_posts = parseLists(neighbor.get("Student posts"))

        neighbor_indicators = getIndicators(neighbor_skills, neighbor_badges, neighbor_bonus, neighbor_quizzes, neighbor_posts)

        '''
          In the following "for cycle" the algorithm will test if this student is able to be classified as a neighbor of the target student.
          To be so, the student must have attempted, at least, the same course features as the target student. This means, for example, that 
          if the target student already has some skills and some badges acquired, in order to be considered neighbor, the other student must 
          have acquired some skills and badges too.

        '''


        count1, count2 = 0, 0
        '''
        for i in range(len(target_student_indicators)):
            if target_student_indicators[i] == 1:
                count1 += 1
                if neighbor_indicators[i] == 1:
                    count2 += 1
        '''

        if neighbor_indicators[0] == 1: # this only works when we are recommending skills from the activity tree

        #if count2 == count1:  # we found a neighbor of the target student

            skills_dic_t, badges_dic_t, bonus_dic_t, quizzes_dic_t = buildActivityXpsDic([neighbor_skills, neighbor_badges, neighbor_bonus, neighbor_quizzes])


            avg_distance_skills, avg_distance_badges, avg_distance_bonus, avg_distance_quizzes, avg_distance_posts = -1, -1, -1, -1, -1


            ##neighbor skills
            new_l = orderElements([k for k in skills_dic], [k for k in skills_dic_t], tag)
            if new_l != []:
                avg_distance_skills = calculateAvgDistance(new_l, tag)

            ##neighbor badges
            new_l = orderElements([k for k in badges_dic], [k for k in badges_dic_t], tag)
            if new_l != []:
                avg_distance_badges = calculateAvgDistance(new_l, tag)

            ##neighbor bonus
            new_l = orderElements([k for k in bonus_dic], [k for k in bonus_dic_t], tag)
            if new_l != []:
                avg_distance_bonus = calculateAvgDistance(new_l, tag)

            ##neighbor quizzes
            new_l = orderElements([k for k in quizzes_dic], [k for k in quizzes_dic_t], tag)
            if new_l != []:
                avg_distance_quizzes = calculateAvgDistance(new_l, tag)

            ##neighbor posts
            new_l = orderElements([k for k in posts], [k for k in neighbor_posts], tag)
            if new_l != []:
                avg_distance_posts = calculateAvgDistance(new_l, tag)

            all_distances = [avg_distance_skills, avg_distance_badges, avg_distance_bonus, avg_distance_quizzes,
                             avg_distance_posts]

            #all_distances = [avg_distance_skills, avg_distance_badges, avg_distance_bonus]

            total_distance = 0
            for el in all_distances:
                if el != -1:
                    total_distance += el

            if total_distance !=0:
                lista_all.append((total_distance,
                                  [("skills", neighbor_skills), ("badges", neighbor_badges), ("bonus", neighbor_bonus),
                                   ("quizzes", neighbor_quizzes), ("posts", neighbor_posts)]))
            else:
                #print("TOTAL DIST ===== 0")
                #print(target_student_indicators, "   VSss   ", neighbor_indicators)
                #print(avg_distance_skills, avg_distance_badges, avg_distance_bonus, avg_distance_quizzes, avg_distance_posts)
                pass
        else:
            #print(target_student_indicators,"   VS   ",neighbor_indicators)
            pass

    return lista_all


def getNeighbors(cluster, file): #
    '''

    :param cluster: target students' cluster
    :param index: get the neighbors of the student which is placed in position "index" from the training file.
    :param file:
    :return: returns the neighbors of the target student that are placed in the profile right above.
    '''

    neighbors = []
    if cluster !=3:
        for dic in file:
            if int(dic.get("cluster")) == cluster+1:
                neighbors.append(dic)
    else:
        for i in range(len(file)):
            if int(file[i].get("cluster")) == cluster:
                neighbors.append(file[i])

    return neighbors


def get_neighbors_alternative(cluster, file):
    '''

    :param cluster:
    :param file:
    :return: returns the neighbors of the target student that are placed in the same profile as the target student.
    '''

    neighbors = []
    for dic in file:
        if int(dic.get("cluster")) == cluster:
            neighbors.append(dic)

    return neighbors


def sortListofTuples(l, tag):
    if tag == "levenshtein" or tag == "jaccard" or tag == "hamming" or tag == "soundex" or tag == "damerau":
        return (sorted(l, key = lambda x: x[0]))
    else:  # jaro and lcs measures
        return reversed(sorted(l, key = lambda x: x[0]))


def aux (activities_list, tag, kNeighbors):
    '''
    :param list_of_tuples: check the param "activities_list" from recommendActivities() method.
    :param tag: could have the following values: "skills", "badges", "bonus", "quizzes", "posts"
    :return:    returns a dictionary where the keys are the activities and the values are Xps that the neighbors obtained in those activities
                Note: this dictionary is ordered by the activities that were more common in the neighbors portfolio, i.e the activities that were performed
                more times
            {"activity1": [350, 120, 220, 400], "activity2":[200,300,100]}
    '''
    dic ={}

    neighbors_considered =0

    for tupl in activities_list:
        if neighbors_considered == kNeighbors:
            break
        all_info = tupl[1]
        for tuple in all_info:
            if tuple[0] == tag:
                if len(tuple[1]) != 0:
                    for el in tuple[1]:
                        if ":" in el:
                            code = el.split(":")[0]
                            xpEarned = el.split(":")[1]
                            if code not in dic:
                                dic[code] = [xpEarned]
                            else:
                                dic[code] += [xpEarned]
                    neighbors_considered+=1              # só os neighbors que tiverem skills feitas irão ser considerados.


    return {k: v for k, v in reversed(sorted(dic.items(), key=lambda item: len(item[1])))}

def firstApproach_auxiliar(all_neighbors_activities, n_recommendations, kneighbors):
    '''
    This function was created only to avoid the code repetition in the "firstApproach" method.

     Ex:
      all_neighbors_activities = {'11906033CompletethePublicistSkill': ['400.0', '200.0'], '11805933CompletetheMoviePosterSkill': ['400.0'],
    '10305333CompletetheCourseLogoSkill': ['100.0'], '09504733CompletetheAlienInvasionSkill': ['100.0']}

        This method returns a dictionary of pairs : ("activity" : Expected Xps).
        This dictionary is ordered by the expected Xps in descending order.
    '''


    act_to_recommend={}

    count = 0

    for key in all_neighbors_activities:
        if count == n_recommendations:
            break
        all_xps = [float(el) for el in all_neighbors_activities.get(key)]
        all_xps.sort(reverse=True)

        #simple avg
        avg = sum(all_xps[0:kneighbors]) / len(all_xps[0:kneighbors])

        #tihs method gives different weights to the neighbors' achieved XPs
        '''
        print(key," - ",all_xps)
        avg = 0
        if len(all_xps) == 1:
            avg = all_xps[0]
        elif len(all_xps) == 2:
            avg = 0.8*all_xps[0] + 0.2*all_xps[1]
        else:
            for i in range(len(all_xps)):
                if i ==0:
                    avg += 0.7 * all_xps[0]
                elif i==1:
                    avg += 0.15 * all_xps[1]
                else:
                    avg += (0.15/(len(all_xps[2:]))) * all_xps[i]
        '''

        act_to_recommend[key] = avg
        count += 1

    return {k: v for k, v in reversed(sorted(act_to_recommend.items(), key=lambda item: item[1]))}



def firstApproach(activities_list, n_recommendations, kneighbors):

    '''
    :param activities_list:  check the param "activities_list" from "recommendActivities" method
    :param n_recommendations:
    :param kneighbors:
    :return:
    '''


    all_neighbors_skills = aux(activities_list, "skills", kneighbors)
    #print("\nall_neighbors_skills: ")
    #print(all_neighbors_skills)
    all_neighbors_badges = aux(activities_list, "badges",kneighbors)

    '''
     Uncomment when you want the algorithm to recommend bonus, quizzes or posts
    
    all_neighbors_bonus = aux(activities_list, "bonus",kneighbors)
    all_neighbor_quizzes = aux(activities_list, "quizzes",kneighbors)
    all_neighbors_posts = aux(activities_list, "posts",kneighbors)
    '''

    skills_to_recommend = firstApproach_auxiliar(all_neighbors_skills, n_recommendations, kneighbors)
    badges_to_recommend = firstApproach_auxiliar(all_neighbors_badges, n_recommendations, kneighbors)

    return [skills_to_recommend, badges_to_recommend]


def recommendActivities(activities_list, kneighbors, n_recommendations):
    '''
    :param activities_list: list of tuples with the following schema:

        [(total_distance, [("skills", neighbor_skills), ("badges", neighbor_badges), ("bonus", neighbor_bonus),
                                   ("quizzes", neighbor_quizzes), ("posts", neighbor_posts)]), (total_distance, [...]) , (total_distance, [...]))

    :param target_student_info:  [student_skills, student_badges, student_bonus, student_quizzes, student_posts]
    :param kneighbors: number of neighbors to have into consideration
    :param n_recommendations: number of recommendations
    :return: list_recommendations is a list of dictionaries where each dictionary represent the skills, badges, bonus, quizzes, posts to be recommended,
        and the values are the average xps earned in each of those activities by the "kneighbors" taken into consideration. The dictionaries are ordered
        from the activities that have more possibilities of maximizing the earned xps by the target student.

        example:

            [{"skill_1":600, "skill_2":550}, {"badge1":350, "badge2":400}, {...}]

    '''

    skills_to_recommend, badges_to_recommend, quizzes_to_recommend, bonus_to_recommend, posts_to_recommend = {},{},{},{},{}

    #these arrays contain all the activities excluding the Xp earned.
    all_neighbors_skills, all_neighbors_badges, all_neighbors_bonus, all_neighbor_quizzes, all_neighbors_posts = {},{},{},{},{}


    '''
    We have two possible strategies in order to recommend some activities to the target student:
    
    1st:
        Recommend those activities which have a bigger representation in the X nearest neighbors.
    2nd:
        Go through the array which contain all the neighbors from the nearest one to the furthest and add activities
        that have not been performed by the target student, until it reaches the desired number of activities to recommend.
        
        Problem of the second approach: If the closest neighbor has 6 activities that were not yet performed by the target student,
        and ig the number of activities to recommend is less or equal to 6, the algorithm will only take into consideration 
        the closest student.  
    '''
    lista = [tupl for tupl in activities_list]

    #print("lista: ")
    #print(lista)


    # 1st approach
    #for i in range(0,kneighbors):
    #    if lista[i]



    list_recommendations = firstApproach(lista, n_recommendations, kneighbors)

    # 2nd approach
    #TODO

    return list_recommendations




def list_occurrences(recommendations):
    code_ocurrences = []  # this list saves the number of ocurrences of each element of the list "recommendations"
    for code in recommendations:
        if code not in [el[0] for el in code_ocurrences]:
            code_ocurrences.append((code, recommendations.count(code)))

    final_list = [ tupl for tupl in reversed(sorted(code_ocurrences, key=lambda item: item[1]))] # sort the list by the number of ocurrences

    return final_list

def drawGraphic(k, tag, all_real_xps, all_expected_xps, colors_labels):
    fig = plt.figure()

    plt.plot([i for i in range(len(all_real_xps))], all_real_xps, colors_labels[1], label = "real Xps")
    plt.plot([i for i in range(len(all_expected_xps))], all_expected_xps, colors_labels[0], label = "expected Xps")

    plt.ylabel("Xps")
    plt.xlabel("Students")
    plt.legend()
    plt.title("graph_k= "+str(k)+" tag= "+tag)
    #plt.show()

    # Create legend & Show graphic
    plt.legend()

    fig.savefig("./results/graph_k= "+str(k)+" tag= "+tag+".png")
    plt.close(fig)

def drawBarChart(improved_students_xps, same_students_xps, worse_students_xps,k,measure):
    fig = plt.figure()

    # set width of bar
    barWidth = 0.25


    # set height of bar
    bars1 = [improved_students_xps[0], improved_students_xps[1], improved_students_xps[2], improved_students_xps[3]]
    bars2 = [same_students_xps[0], same_students_xps[1], same_students_xps[2], same_students_xps[3]]
    bars3 = [worse_students_xps[0], worse_students_xps[1], worse_students_xps[2], worse_students_xps[3]]

    # Set position of bar on X axis
    r1 = [i for i in range(len(bars1))]
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#557f2d', width=barWidth, edgecolor='white', label='Increases')
    plt.bar(r2, bars2, color='#2d7f5e', width=barWidth, edgecolor='white', label='Maintains')
    plt.bar(r3, bars3, color='#7f6d5f', width=barWidth, edgecolor='white', label='Decreases')

    # Add xticks on the middle of the group bars
    plt.xlabel('Clusters', fontweight='bold')
    plt.ylabel("Nº of Students", fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Underachievers', 'Half Hearted', 'Regular', 'Achievers'])

    plt.title("System Performance \n K= " + k + "  SM= " + measure)

    # Create legend & Show graphic
    plt.legend()

    fig.savefig("./results/barChart"+k+"_"+measure+ ".png")
    plt.close(fig)

def drawBoxPlot(u,h,r,ac,tag,k,measure):


    fig = plt.figure()


    a = pd.DataFrame({'group': np.repeat('U', len(u)), 'value': u})
    b = pd.DataFrame({'group': np.repeat('H', len(h)), 'value': h })
    c = pd.DataFrame({'group': np.repeat('R', len(r)), 'value': r})
    d = pd.DataFrame({'group': np.repeat('A', len(ac)), 'value': ac})
    df = a.append(b).append(c).append(d)


    # Usual boxplot
    sns.boxplot(x='group', y='value', data=df)



    ax = sns.boxplot(x='group', y='value', data=df)
    #ax = sns.stripplot(x='group', y='value', data=df, color="orange", jitter=0.2, size=2.5)

    # Calculate number of obs per group & median to position labels
    #medians = df.groupby(['group'])['value'].median().values
    all = [u, h, r, ac]
    for l in all:
        l.sort()
    medians = [median(l) for l in all if l!=[]]
    nobs = ["n: "+ str(len(l)) for l in [u,h,r,ac] if l!=[]] #number of observations

    # Add it to the plot
    pos = range(len(nobs))
    for tick, label in zip(pos, ax.get_xticklabels()):
        plt.text(pos[tick], medians[tick] + 0.4, nobs[tick], horizontalalignment='center', size='medium', color='black',
                 weight='semibold')

    plt.ylabel("XPs")

    if tag == "after":
        plt.title("XP earned by cluster following System Recommendations\n K= "+k+"  SM= "+measure)
        fig.savefig("./results/boxPlot_" + k + "_" + measure + ".png")
    else:
        plt.title("XP earned by cluster without following System Recommendations")
        fig.savefig("./results/boxPlot_before.png")

    #plt.show()
    plt.close(fig)


def smartStudentDistributionPerCluster(dic_all_clusters):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    names = []
    data = []
    for key in dic_all_clusters:
        names.append(key)
        data.append(dic_all_clusters[key])

    total_students = sum(data)
    for i in range(len(data)):
        names[i] = str(names[i]) + "( "+str(data[i])+")"


    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(names[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    #ax.set_title("Data distribution over the years")

    plt.show()



def drawTrainAndTestPopulation(input):

    dic_all_clusters = {"UnderAchievers": 0, "Half-Hearted": 0, "Regular": 0, "Achievers": 0}

    for i in range(len(input)):
        student_cluster = int(input[i].get("cluster"))
        if student_cluster ==0:
            dic_all_clusters["UnderAchievers"] += 1
        elif student_cluster ==1:
            dic_all_clusters["Half-Hearted"] += 1
        elif student_cluster ==2:
            dic_all_clusters["Regular"] += 1
        else:
            dic_all_clusters["Achievers"] += 1

    smartStudentDistributionPerCluster(dic_all_clusters)


def drawPerformanceOverK(dic_subiu, dic_desceu, dic_manteve, tag):
    fig = plt.figure()

    x1,y1 = [k for k in dic_subiu], [dic_subiu[k] for k in dic_subiu]
    ymax = max(y1)
    xpos = y1.index(ymax)
    xmax = x1[xpos]
    plt.plot(x1,y1, 'g', label="Increases")
    plt.annotate('Max at '+str(xmax), xy=(xmax, ymax), xytext=(xmax, ymax + 5),
                arrowprops=dict(facecolor='darksalmon', shrink=0.05),
                )

    # ///////////////////////////////////////////////////
    x2, y2 = [k for k in dic_subiu], [dic_desceu[k] for k in dic_desceu]

    ymin = min(y2)
    xpos2 = y2.index(ymin)
    xmin = x2[xpos2]

    plt.plot(x2,y2, 'red', label="Decreases")
    plt.annotate('Min at ' + str(xmin), xy=(xmin, ymin), xytext=(xmin, ymin + 5),
                 arrowprops=dict(facecolor='darksalmon', shrink=0.05),
                 )

    #////////////////////////////////////////////////////

    plt.plot([k for k in dic_subiu], [dic_manteve[k] for k in dic_manteve], 'blue', label="Maintains")


    plt.xlabel("K Neighbors")
    plt.ylabel("Nº of Students")
    plt.legend()
    plt.title("Performance Over K neighbors\n SM= " + tag)
    # plt.show()

    # Create legend & Show graphic
    plt.legend()

    fig.savefig("./results/Performance_Over_K_neighbors_SM= " + tag + ".png")
    plt.close(fig)

def main():
    #date_range = "/03/15"
    # date_range = "/04/15"
    date_range = "/05/20"

    test_year = 2018

    data = date_range[1:].split("/")
    l = readCSVfile("./train&test_files/"+data[1]+"_"+data[0]+"/trainSet_" + str(test_year) + "_dateRange=" +data[1]+"_"+data[0] + ".csv", ",")
    train_input, train_output = buildInputandOutput(l)

    l2 = readCSVfile("./train&test_files/"+data[1]+"_"+data[0]+"/testSet_" + str(test_year) + "_dateRange=" +data[1]+"_"+data[0] + ".csv", ",")
    test_input, test_output = buildInputandOutput(l2)

    drawTrainAndTestPopulation(train_input)
    drawTrainAndTestPopulation(test_input)

    all_measures =["soundex", "jaccard", "jaro", "damerau"]
    #all_measures = ["damerau"]
    colors_labels = ['g', 'red']
    colors_labels2 = ['yellowgreen', 'violet', 'tan', 'maroon']

    xps_chosen_student=[] # xps earned by the student without following the system recommendationss

    for tag in all_measures:
        performance_over_k_increases,performance_over_k_decreases,performance_over_k_maintains = {},{},{}

        print("tag: ",tag)
        #accuracy_k =[]

        for k in range(1,16): # k neighbors
            acertou, errou = 0, 0
            print("K= ",k)
            all_real_xps = []
            all_expected_xps =[]
            improved_students_xps, same_students_xps, worse_students_xps =[0,0,0,0], [0,0,0,0], [0,0,0,0]

            under_before_xps, half_before_xps, reg_before_xps, ach_before_xps = [],[],[],[]
            under_after_xps, half_after_xps, reg_after_xps, ach_after_xps = [],[],[],[]

            dic_all_clusters={"UnderAchievers":0,"Half-Hearted":0,"Regular":0,"Achievers":0}
            for i in range (len(test_input)):

                student_cluster = int(test_input[i].get("cluster"))
                student_skills = parseLists(test_input[i].get("Student skills"))
                student_badges = parseLists(test_input[i].get("Student badges"))
                student_bonus = parseLists(test_input[i].get("Student bonus"))
                student_quizzes = parseLists(test_input[i].get("Student quizzes"))
                student_posts = parseLists(test_input[i].get("Student posts"))


                if student_skills ==[] and student_badges ==[] and student_bonus ==[] and student_quizzes==[] and student_posts==[]:
                    print("TUDO vazio")
                    continue

                n = getNeighbors(student_cluster, train_input)

                #n= get_neighbors_alternative(student_cluster, train_input)



                lista_all = scrutinizeData(n, student_skills, student_badges, student_quizzes, student_bonus, student_posts, tag)

                if lista_all == []:
                    print("student_cluster: ",student_cluster)
                    print("student_skills: ",student_skills)
                    print("student_badges: ",student_badges)
                    print("student_bonus: ",student_bonus)
                    print("student_quizzes: ",student_quizzes)
                    print("student_posts: ",student_posts)


                # this list contains all the neighbor's activities of the target student ordered by
                # the distance that each of them is from the target student
                lista = sortListofTuples(lista_all, tag)


                next_skills_student = parseLists(test_output[i].get("Next Skills"))
                next_badges_student = parseLists(test_output[i].get("Next badges"))
                next_bonus_student = parseLists(test_output[i].get("Next bonus"))



                list_all_recommendations = recommendActivities(lista, k, 3)


                skills_recommended_by_the_system = list_all_recommendations[0]


                if skills_recommended_by_the_system!={} :
                    earned_xps = 0

                    # next_badges = next_badges_student[1]
                    # next_bonus = next_bonus_student[2]

                    # the dictionary is ordered from the skills that are expected to maximize the earned Xps by the target student
                    best_skill_to_recommend = list(skills_recommended_by_the_system.keys())[0]
                    expected_xps = skills_recommended_by_the_system.get(best_skill_to_recommend)

                    if next_skills_student != []:

                        skill_chosen_by_the_student = next_skills_student[0].split(":")[0]
                        earned_xps = float(next_skills_student[0].split(":")[1])


                    all_real_xps.append(earned_xps)
                    all_expected_xps.append(expected_xps)



                    if expected_xps> earned_xps:
                        #subiu
                        improved_students_xps[student_cluster] +=1

                    elif expected_xps == earned_xps:
                        #manteve
                        same_students_xps[student_cluster] +=1
                    else:
                        #desceu
                        worse_students_xps[student_cluster]+=1

                    if student_cluster ==0:
                        under_before_xps.append(earned_xps)
                        under_after_xps.append(expected_xps)
                        dic_all_clusters["UnderAchievers"] +=1
                    elif student_cluster==1:
                        half_before_xps.append(earned_xps)
                        half_after_xps.append(expected_xps)
                        dic_all_clusters["Half-Hearted"]+=1

                    elif student_cluster==2:
                        reg_before_xps.append(earned_xps)
                        reg_after_xps.append(expected_xps)
                        dic_all_clusters["Regular"]+=1

                    else:
                        ach_before_xps.append(earned_xps)
                        ach_after_xps.append(expected_xps)
                        dic_all_clusters["Achievers"]+=1

                else:
                    print ("\nCluster: "+str(student_cluster))
                    print("next_skills_student :")
                    print(next_skills_student)
                    print("skills_recommended_by_the_system : ")
                    print(skills_recommended_by_the_system)
                    print("Neighbors: ")
                    for nei in n:
                        print(nei)
                    print("lista_all")
                    print(lista_all)
                    print("lista_all_recommendations: ")
                    print(list_all_recommendations)

            performance_over_k_increases[k] = sum(improved_students_xps)
            performance_over_k_decreases[k] = sum(worse_students_xps)
            performance_over_k_maintains[k] =sum(same_students_xps)


            #drawBoxPlot(under_before_xps, half_before_xps, reg_before_xps, ach_before_xps, "before", str(k), tag)
            #drawBoxPlot( under_after_xps, half_after_xps, reg_after_xps, ach_after_xps,"after", str(k), tag)
            #drawBarChart(improved_students_xps, same_students_xps,worse_students_xps,str(k),tag)

        #drawPerformanceOverK(performance_over_k_increases,performance_over_k_decreases,performance_over_k_maintains, tag)


