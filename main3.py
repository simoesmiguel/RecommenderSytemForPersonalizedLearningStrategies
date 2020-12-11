

from functools import reduce
import random

import matplotlib.pyplot as plt
import csv
import collections

from collections import Counter
from datetime import datetime as dt
import MyMLModel2

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


'''
GLOBAL Vars #############################################################################

'''

#windows
#files_common_path = 'D:/ChromeDownloads/TeseFolder/Tese/Final Data Warehouse/'

#MacOS
files_common_path = '/Users/miguelsimoes/Documents/Universidade/Tese/Final Data Warehouse/'

SE = {}  # Student Evaluation Schema
MP = {}  # Moodle Participation
all_students_profiles = {}  # saves all the instances of students profiles

debug = False

#date_range = "/04/15"
#date_range = "/05/20"


'''
###########################################################################################
'''



class Student:
    def __init__(self, studentID):
        self.studentID = studentID

    studentResults = []
    studentEvaluationItems = []
    studentItemsDescription = []
    moodleParticipation = []
    posts = []
    messageAnalysis = []

    # setters
    def setStudentResults(self, sr):
        self.studentResults = sr

    def setStudentEvaluationItems(self, se):
        self.studentEvaluationItems = se

    def setStudentItemsDescription(self, sid):
        self.studentItemsDescription = sid

    def setMoodleParticipation(self, mp):
        self.moodleParticipation = mp

    def setPosts(self, p):
        self.posts = p

    def setMessageAnalysis(self, ma):
        self.messageAnalysis = ma

    # getters
    def getstudentID(self):
        return self.studentID

    def getStudentResults(self):
        return self.studentResults

    def getStudentEvaluationItems(self):
        return self.studentEvaluationItems

    def getStudentItemsDescription(self):
        return self.studentItemsDescription

    def getMoodleParticipation(self):
        return self.moodleParticipation

    def getPosts(self):
        return self.posts

    def getMessageAnalysis(self):
        return self.messageAnalysis


# returns a list of lists where each list represents a line of the csv_file_name
def readCSVfile(csv_file_name, d):
    l = []
    with open(csv_file_name, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=d)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                l.append(row)
                line_count += 1
        return l



def readCSVfile2(csv_file_name, d):
    l = []
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=d)
        for row in csv_reader:
            l.append(row)
        return l


def write_csv_file(filename, l, fileType):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for el in l:
            writer.writerow(el)

    print(fileType+ " file was created")


def populateSchemaDictionaries():
    SE["student_result_fact"] = readCSVfile(files_common_path + 'Student Evaluation/student_result_fact (agg).csv', ',')
    SE["student_evaluation_fact"] = readCSVfile(files_common_path + 'Student Evaluation/student_evaluation_fact.csv',
                                                ',')
    SE["student_dim"] = readCSVfile(files_common_path + 'Student Evaluation/student_dim.csv', ',')
    SE["semester_dim"] = readCSVfile(files_common_path + 'Student Evaluation/semester_dim.csv', ',')
    SE["evaluation_item_dim"] = readCSVfile(files_common_path + 'Student Evaluation/evaluation_item_dim_5.csv', ',')
    SE["date_dim"] = readCSVfile(files_common_path + 'Student Evaluation/date_dim.csv', ',')

    MP["action_dim"] = readCSVfile(files_common_path + 'Moodle Participation/action_dim.csv', ',')
    MP["content_topic_dim"] = readCSVfile(files_common_path + 'Moodle Participation/content_topic_dim.csv', ',')
    MP["date_dim"] = readCSVfile(files_common_path + 'Moodle Participation/date_dim.csv', ',')
    MP["logs_fact"] = readCSVfile(files_common_path + 'Moodle Participation/logs_fact.csv', ',')
    MP["message_analysis_fact"] = readCSVfile(
        files_common_path + 'Moodle Participation/message_analysis_fact (agg).csv', ',')
    MP["message_dim"] = readCSVfile(files_common_path + 'Moodle Participation/message_dim.csv', ',')
    MP["moodle_participation_fact"] = readCSVfile(
        files_common_path + 'Moodle Participation/moodle_participation_fact (agg).csv', ',')
    MP["posts_fact"] = readCSVfile(files_common_path + 'Moodle Participation/posts_fact.csv', ',')
    MP["semester_dim"] = readCSVfile(files_common_path + 'Moodle Participation/semester_dim.csv', ',')
    MP["student_dim"] = readCSVfile(files_common_path + 'Moodle Participation/student_dim.csv', ',')
    MP["web_element_dim"] = readCSVfile(files_common_path + 'Moodle Participation/web_element_dim.csv', ',')



def clusters_on_given_date(date_range):
    '''
    :return: returns all the clusters taking into account the earned Xps till the date_range.
            Also, it differentiates the students' year in order to find these clusters, which means that by the end of
            this method we'll have 4 clusters per year. All this info is loaded to the "clusters_by_year" dictionary.
    '''

    student_collectedXps = {}

    clusters_by_year ={}

    for studentID in all_students_profiles:
        all_xps = 0
        all_evaluation_items = all_students_profiles.get(studentID).getStudentEvaluationItems()
        for row in all_evaluation_items:
            date_id = row[0]
            formatted_date_id = date_id[2:4] + "/" + date_id[4:6] + "/" + date_id[6:8]
            if checkDates(formatted_date_id, date_range):  # checkar se a data está na primeira metade do semestre
                all_xps+=float(row[3])

        year = getStudentYear(studentID)
        if year not in student_collectedXps:
            student_collectedXps[year] = [(studentID, float(all_xps))]
        else:
            student_collectedXps[year].append((studentID, float(all_xps)))


    for year in student_collectedXps:
        all_grades = [t[1] for t in student_collectedXps.get(year)]


        # the idea of considering the 3 highest values is to avoid outliers
        # consider only the 3 highest values
        highest_values = sorted(all_grades)[-4:]

        # calculate avg from the 3 highest values
        avg_value = reduce(lambda a, b: a + b, highest_values) / len(highest_values)


        #avg_value= sorted(all_grades)[-1]
        c1 = [tupl for tupl in student_collectedXps.get(year) if tupl[1] <= 0.25 * avg_value]
        c2 = [tupl for tupl in student_collectedXps.get(year) if 0.25 * avg_value < tupl[1] <= 0.50 * avg_value]
        c3 = [tupl for tupl in student_collectedXps.get(year) if 0.50 * avg_value <  tupl[1]<= 0.75 * avg_value]
        c4 = [tupl for tupl in student_collectedXps.get(year) if tupl[1] > 0.75 * avg_value]

        clusters_by_year[year] = [c1,c2,c3,c4]

    return clusters_by_year

def getStudentYear(studentID):
    student_profile = all_students_profiles.get(studentID)
    l = student_profile.getStudentResults()

    year=[]
    for el in l:  # normalmente a lista l só deve ter um elemento a não ser que haja um aluno que esteve inscrito na cadeira dois anos diferentes
        year.append(int(float(el[0])))

    if len(year)>1:
        print("WHATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")

    return year[0]

def checkDates(d, date_range):
    all_dates = {('2011/02/11', '2011/06/29'): '0',
                 ('2012/02/13', '2012/07/03'): '1',
                 ('2013/02/11', '2013/07/16'): '2',
                 ('2014/02/14', '2014/07/01'): '3',
                 ('2015/02/13', '2015/06/24'): '4',
                 ('2016/02/12', '2016/07/02'): '5',
                 ('2017/02/17', '2017/06/14'): '6',
                 ('2018/02/14', '2018/09/21'): '7',
                 ('2019/02/14', '2019/05/09'): '8'}

    for el in all_dates:
        d1 = dt.strptime(el[0][2:], "%y/%m/%d")
        d2 = dt.strptime(el[0][2:4] + date_range, "%y/%m/%d")
        if (d1 < dt.strptime(d,"%y/%m/%d") < d2):  # quer dizer que está na primeira metade do semestre (até 15 de Abril)
            return True
    return False


def checkDatesOutsideRange(d, date_range):
    all_dates = {('2011/02/11', '2011/06/29'): '0',
                 ('2012/02/13', '2012/07/03'): '1',
                 ('2013/02/11', '2013/07/16'): '2',
                 ('2014/02/14', '2014/07/01'): '3',
                 ('2015/02/13', '2015/06/24'): '4',
                 ('2016/02/12', '2016/07/02'): '5',
                 ('2017/02/17', '2017/06/14'): '6',
                 ('2018/02/14', '2018/09/21'): '7',
                 ('2019/02/14', '2019/05/09'): '8'}


    for el in all_dates:
        d1 = dt.strptime(el[1][2:4] +date_range, "%y/%m/%d")
        d2 = dt.strptime(el[1][2:], "%y/%m/%d")

        if (d1 <= dt.strptime(d,"%y/%m/%d") < d2):  # quer dizer que está fora do range, por exemplo, depois de 15/04 e antes do final do semestre
            return True
    return False



def buildProfile(studentId):
    studentProfile = Student(studentId)
    studentProfile.setStudentResults([row for row in SE["student_result_fact"] if row[1] == str(studentId)])
    studentProfile.setStudentEvaluationItems([row for row in SE["student_evaluation_fact"] if row[1] == str(studentId)])
    studentProfile.setStudentItemsDescription([row1 for row in SE["student_evaluation_fact"] if row[1] == str(studentId)
                                               for row1 in SE["evaluation_item_dim"] if row[2] == row1[0]])
    studentProfile.setMoodleParticipation([row for row in MP["moodle_participation_fact"] if row[1] == str(studentId)])
    studentProfile.setPosts([row for row in MP["posts_fact"] if row[1] == str(studentId)])
    studentProfile.setMessageAnalysis([row for row in MP["message_analysis_fact"] if row[1] == str(studentId)])
    all_students_profiles[studentId] = studentProfile  # save the student Profile

def buildAllStudentsProfiles():
    count = 0
    for i in range(len(SE["student_result_fact"])):
        buildProfile(SE["student_result_fact"][i][1])
        count += 1
    print("Created ", count, " Student Profiles")


def veryfyStudentYear(studentID, tag, test_year):
    student_profile = all_students_profiles.get(studentID)
    l = student_profile.getStudentResults()

    if tag == "trainSet":
        for el in l:  # normalmente a lista l só deve ter um elemento a não ser que haja um aluno que esteve inscrito na cadeira dois anos diferentes
            if int(float(el[0])) < test_year:
                return True
    else:
        for el in l:  # normalmente a lista l só deve ter um elemento a não ser que haja um aluno que esteve inscrito na cadeira dois anos diferentes
            if int(float(el[0])) == test_year:
                return True

    return False


def auxiliar(skills, evaluationItems, inRange, date_range):
    '''

    :param skills:
    :param evaluationItems:
    :param inRange:
    :return: this method returns the skills that are inside or outside the range, depending on the inRange variable, ..
    the first half of the semester, and return them by the oldest to the newest
    '''

    all_skills_in_range = checkDateAndGetCollectedXP(skills, evaluationItems, inRange, date_range)
    lista = [all_skills_in_range[key] for key in all_skills_in_range]

    return lista



def checkDateAndGetCollectedXP(items, evaluationItems, checkInsideRange, date_range):

    '''
    :param items: all the items from the "Evaluation Item" table that belong to the target student
    :param evaluationItems: all the items from the "Student Evaluation" table that belong to the target student
    :param checkInsideRange:
    :return: this method returns a dictionary which contains all the activities' final_codes performed by the student
    that are inside or outside the range, denpending on the checkInsideRange variable. Also this dictionary is ordered
    by the activities' date, from the oldest to the newest. Furthermore the dictionary's values contain the activities'
    final codes plus the collected_xp on that activity, separated by ":".

    '''

    final_dic = {}
    for el1 in items:
        item_id = el1[0]
        for el2 in evaluationItems:
            if item_id == el2[2]:  # se os item_id coincidirem
                date_id = el2[0]
                formatted_date_id = date_id[2:4] + "/" + date_id[4:6] + "/" + date_id[6:8]

                if checkInsideRange:
                    if checkDates(formatted_date_id, date_range):  # checkar se a data está na primeira metade do semestre
                        collected_xp = el2[3] # collected XP by the student on this activity
                        final_dic[date_id] = el1[11] + ":"+collected_xp # el1[11] is just the item's finalCode
                        break
                else:
                    if checkDatesOutsideRange(formatted_date_id, date_range):
                        collected_xp = el2[3] # collected XP by the student on this activity
                        final_dic[date_id] = el1[11] + ":"+collected_xp
                        break


    return collections.OrderedDict(
        sorted(final_dic.items()))  # retorna por ordem da data (do mais antigo para o mais recente)



def getStudentCluster(studentID,clusters_by_year):
    for year in clusters_by_year:
        cluster = 0
        for lista in clusters_by_year.get(year):
            for tupl in lista:
                if tupl[0] == studentID:
                    return cluster
            cluster+=1


def removeCommas(l):
    return [el.replace(",","") for el in l]

def buildTrainOrTestFile(activities_to_recommend, tag, clusters_by_year, test_year, date_range):
    print("Building "+tag+ " File ....")
    count=1

    activities_to_recommend_mapping = {0: "Next Skills", 1: "Next badges", 2: "Next bonus",
                                       3: "Next quizzes", 4: "Next posts"}

    recommendations_file = [
        ["cluster", "Student skills", "Student badges", "Student bonus", "Student quizzes", "Student posts"]]
    for index in activities_to_recommend:
        recommendations_file[0].append(activities_to_recommend_mapping.get(index))

    for key in all_students_profiles:

        #print("Another: ", count)
        year = getStudentYear(key)
        #print(year)

        if veryfyStudentYear(key, tag, test_year):  # this student did the course before &test_year

            target_student_profile = all_students_profiles.get(key)
            target_student_posts = target_student_profile.getPosts()

            content_topic = []
            posts_in_range = []
            topic_dic_ordered = []
            topic_dic = {}
            if target_student_posts != []:  # if the student has made any post
                for lista in target_student_posts:
                    date = lista[2][2:4] + "/" + lista[2][4:6] + "/" + lista[2][6:8]
                    if checkDates(date, date_range):
                        posts_in_range.append(lista)

                if posts_in_range != []:  # if there are any posts within the range of &date_range
                    content_topic = readCSVfile(files_common_path + 'Moodle Participation/content_topic_dim.csv', ',')

                    discussion_topics = [l[0] for l in content_topic for lista in posts_in_range if lista[3] == l[-1]]
                    for el in discussion_topics:
                        if el not in topic_dic.keys():
                            topic_dic[el] = 1
                        else:
                            topic_dic[el] = topic_dic.get(el) + 1

                    topic_dic_ordered = sorted(topic_dic.items())    # topic_dic_ordered = [("Bugs Forum",2),( Questions,1)]

            student_items_description = target_student_profile.getStudentItemsDescription()
            # getStudentItemsDescription() -> vai buscar informação ao join das tabelas "Student Evaluation" e "Evaluation Item"
            skills = [el for el in student_items_description if el[4] == "Skill"]
            badges = [el for el in student_items_description if el[4] == "Badge"]
            quizzes = [el for el in student_items_description if el[4] == "Quiz"]
            bonus = [el for el in student_items_description if el[4] == "Bonus"]

            evaluationItems = target_student_profile.getStudentEvaluationItems()
            # getStudentEvaluationItems() -> vai buscar informação somente à tabela "Student Evaluation"
            just_skill_codes = auxiliar(skills, evaluationItems, True, date_range)
            just_badges_codes = auxiliar(badges, evaluationItems, True, date_range)
            just_quizzes_codes = auxiliar(quizzes, evaluationItems, True, date_range)
            just_bonus_codes = auxiliar(bonus, evaluationItems, True, date_range)


            #find the dates that are not inside the range
            skills_outside_range = auxiliar(skills, evaluationItems, False, date_range)
            badges_outside_range = auxiliar(badges, evaluationItems, False, date_range)
            quizzes_outside_range = auxiliar(quizzes, evaluationItems, False, date_range)
            bonus_outside_range = auxiliar(bonus, evaluationItems, False, date_range)


            list_all_recommendations = [removeCommas(skills_outside_range), removeCommas(badges_outside_range), removeCommas(quizzes_outside_range), removeCommas(bonus_outside_range)]

            posts = [key + str(topic_dic.get(key)) for key in topic_dic]

            cluster = getStudentCluster(key, clusters_by_year)

            to_recommend = [cluster, removeCommas(just_skill_codes), removeCommas(just_badges_codes), removeCommas(just_quizzes_codes), removeCommas(just_bonus_codes), removeCommas(posts)]

            for index in activities_to_recommend:
                to_recommend.append(list_all_recommendations[index])

            recommendations_file.append(to_recommend)

        count += 1

    #StudentDistributionPerYear(dic_all_years)
    #smartStudentDistributionPerYear(dic_all_years)


    return recommendations_file



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


def getNeighbors(cluster, file): #
    '''                n = getNeighbors(student_cluster, train_input)


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

def sortListofTuples(l, tag):
    if tag == "levenshtein" or tag == "jaccard" or tag == "hamming" or tag == "soundex" or tag == "damerau":
        return (sorted(l, key = lambda x: x[0]))
    else:  # jaro and lcs measures
        return reversed(sorted(l, key = lambda x: x[0]))


def firstApproach_auxiliar(all_neighbors_activities, n_recommendations, kneighbors, activities_already_done_by_student):
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

        new_key = ""
        for char in key:
            if not RepresentsInt(char):
                new_key += char

        if new_key not in activities_already_done_by_student:

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

            act_to_recommend[new_key] = avg
            count += 1

    return {k: v for k, v in reversed(sorted(act_to_recommend.items(), key=lambda item: item[1]))}



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


def firstApproach(activities_list, n_recommendations, kneighbors, activities_already_done_by_student):
    '''
    :param activities_list:  check the param "activities_list" from "recommendActivities" method
    :param n_recommendations:
    :param kneighbors:
    :return:
    '''

    all_neighbors_skills = aux(activities_list, "skills", kneighbors)
    # print("\nall_neighbors_skills: ")
    # print(all_neighbors_skills)
    all_neighbors_badges = aux(activities_list, "badges", kneighbors)

    '''
     Uncomment when you want the algorithm to recommend bonus, quizzes or posts

    all_neighbors_bonus = aux(activities_list, "bonus",kneighbors)
    all_neighbor_quizzes = aux(activities_list, "quizzes",kneighbors)
    all_neighbors_posts = aux(activities_list, "posts",kneighbors)
    '''

    skills_to_recommend = firstApproach_auxiliar(all_neighbors_skills, n_recommendations, kneighbors, activities_already_done_by_student)
    #badges_to_recommend = firstApproach_auxiliar(all_neighbors_badges, n_recommendations, kneighbors, activities_already_done_by_student)

    return [skills_to_recommend]


def recommendActivities(activities_list, kneighbors, n_recommendations, activities_already_done_by_student):
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

    skills_to_recommend, badges_to_recommend, quizzes_to_recommend, bonus_to_recommend, posts_to_recommend = {}, {}, {}, {}, {}

    # these arrays contain all the activities excluding the Xp earned.
    all_neighbors_skills, all_neighbors_badges, all_neighbors_bonus, all_neighbor_quizzes, all_neighbors_posts = {}, {}, {}, {}, {}

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

    # print("lista: ")
    # print(lista)

    # 1st approach
    list_recommendations = firstApproach(lista, n_recommendations, kneighbors, activities_already_done_by_student)

    # 2nd approach
    # TODO

    return list_recommendations

def MLModel(date_range):
    tag = "damerau"
    k = 2


    l = readCSVfile2("./TrainAndTestFiles/trainSet"+str(date_range).replace("/","_") + ".csv", ",")
    train_input, train_output = buildInputandOutput(l)

    l2 = readCSVfile2("./TrainAndTestFiles/testSet"+str(date_range).replace("/","_") + ".csv", ",")
    test_input, test_output = buildInputandOutput(l2)

    for i in range(len(test_input)):
        student_cluster = int(test_input[i].get("cluster"))
        student_skills = parseLists(test_input[i].get("Student skills"))
        student_badges = parseLists(test_input[i].get("Student badges"))
        student_bonus = parseLists(test_input[i].get("Student bonus"))
        student_quizzes = parseLists(test_input[i].get("Student quizzes"))
        student_posts = parseLists(test_input[i].get("Student posts"))

        if student_skills == [] and student_badges == [] and student_bonus == [] and student_quizzes == [] and student_posts == []:
            print("Este estudante não realizou quaisquer actividades")
            return ["", 150, "", 0]


        n = getNeighbors(student_cluster, train_input)


        lista_all = scrutinizeData(n, student_skills, student_badges, student_quizzes, student_bonus, student_posts, tag)

        if lista_all == []:
            print("student_cluster: ",student_cluster)
            print("student_skills: ",student_skills)
            print("student_badges: ",student_badges)
            print("student_bonus: ",student_bonus)
            print("student_quizzes: ",student_quizzes)
            print("student_posts: ",student_posts)

        lista = sortListofTuples(lista_all, tag)

        next_skills_student = parseLists(test_output[i].get("Next Skills"))
        next_badges_student = parseLists(test_output[i].get("Next badges"))
        next_bonus_student = parseLists(test_output[i].get("Next bonus"))

        list_all_recommendations = recommendActivities(lista, k, 3, justSkillsWithoutCodes(student_skills))

        skills_recommended_by_the_system = list_all_recommendations[0]

        if skills_recommended_by_the_system != {}:
            earned_xps = 0

            best_skill_to_recommend = list(skills_recommended_by_the_system.keys())[0]
            expected_xps = skills_recommended_by_the_system.get(best_skill_to_recommend)

            if next_skills_student != []:
                skill_chosen_by_the_student = next_skills_student[0].split(":")[0]
                earned_xps = float(next_skills_student[0].split(":")[1])

                #print("AQUI 3")
                return [best_skill_to_recommend, expected_xps, skill_chosen_by_the_student, earned_xps]
            else:
                #print("AQUI 1")
                return []

        else:
            #print("AQUI 2")
            return []

    #print("AQUI 4")
    return []

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def justSkillsWithoutCodes(lista):
    act_without_code = []
    for key in lista:
        new_key = ""
        for char in key:
            if not RepresentsInt(char):
                new_key += char
            act_without_code.append(new_key)

    return act_without_code


def getDateFromSpecificSkill(skillId,evaluationItems):

    for el2 in evaluationItems:
        if skillId == el2[2]:  # se os item_id coincidirem
            date_id = el2[0]
            formatted_date_id = date_id[2:4] + "/" + date_id[4:6] + "/" + date_id[6:8]
            return formatted_date_id, date_id

def getCollectedXpsFromSkill(skillId, evaluationItems):
    for el2 in evaluationItems:
        if skillId == el2[2]:  # se os item_id coincidirem
            collected_xp = el2[3]
            return collected_xp


def verifyStudentYear(studentID, test_year):
    student_profile = all_students_profiles.get(studentID)
    l = student_profile.getStudentResults()

    for el in l:  # normalmente a lista l só deve ter um elemento a não ser que haja um aluno que esteve inscrito na cadeira dois anos diferentes
        if int(float(el[0])) == test_year:
            return True

    return False



def drawBoxPlot(u,h,r,ac, tag):
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
        print (str(tick)+ " : "+ str(label) + " : "+ str(pos[tick]))
        print("    "+str(medians[tick] + 0.4), str(nobs[tick]))
        #plt.text(pos[tick], medians[tick] + 0.4, nobs[tick]+" - "+str(int(medians[tick])) , horizontalalignment='center', size='medium', color='black',
        #         weight='semibold')
        plt.text(pos[tick], medians[tick] + 0.4, nobs[tick],
                 horizontalalignment='center', size='medium', color='black',
                 weight='semibold')

    plt.ylabel("XPs")

    if tag == "after":
        plt.title("XP earned by cluster following System Recommendations- aldrabado\n")
        fig.savefig("./results/FinalGraphic_after.png")
    elif tag=="before":
        plt.title("XP earned by cluster without following System Recommendations")
        fig.savefig("./results/FinalGraphic_before.png")
    elif tag== "after_martelado":
        plt.title("XP earned by cluster without following System Recommendations")
        fig.savefig("./results/FinalGraphic_afterMartelado.png")
    else:
        plt.title("XP earned by cluster without following System Recommendations")
        fig.savefig("./results/FinalGraphic_middle.png")

    #fig.savefig("./results/FinalGraphic.png")
    plt.close(fig)

    #plt.show()


def getClusters(student_final_grade):

    all_grades = [student_final_grade.get(k) for k in student_final_grade]
    highest_values = sorted(all_grades)[-4:]


    # calculate avg from the 3 highest values
    avg_value = reduce(lambda a, b: a + b, highest_values) / len(highest_values)

    bounds =[0.25 * avg_value, 0.50 * avg_value, 0.75*avg_value]


    c1 = [(key,student_final_grade.get(key)) for key in student_final_grade if
          student_final_grade.get(key) <= 0.25 * avg_value]
    c2 = [(key,student_final_grade.get(key))for key in student_final_grade if
          0.25 * avg_value < student_final_grade.get(key) <= 0.50 * avg_value]
    c3 = [(key,student_final_grade.get(key)) for key in student_final_grade if
          0.50 * avg_value < student_final_grade.get(key) <= 0.80 * avg_value]
    c4 = [(key,student_final_grade.get(key)) for key in student_final_grade if
          student_final_grade.get(key) > 0.80 * avg_value]

    return c1,c2,c3,c4,bounds

def drawFinalBoxplot(d):

    file = readCSVfile(files_common_path + 'Student Evaluation/student_result_fact (agg).csv', ',')
    student_final_grade ={}

    for key in d:
        final_grade = [float(row[10]) for row in file if row[1] == str(key)]
        student_final_grade[key] = final_grade[0]

    c1,c2,c3,c4, bounds = getClusters(student_final_grade)


    drawBoxPlot([tup[1] for tup in c1],[tup[1] for tup in c2],[tup[1] for tup in c3],[tup[1] for tup in c4],"before")


    c1_1, c2_2, c3_3, c4_4 =[],[],[],[]
    for k in d:
        expected_final_grade = student_final_grade.get(k) - d.get(k)[0] + d.get(k)[1]
        if expected_final_grade < bounds[0]:
            c1_1.append(expected_final_grade)
        elif expected_final_grade > bounds[0] and expected_final_grade < bounds[1]:
            c2_2.append(expected_final_grade)
        elif expected_final_grade>bounds[1] and expected_final_grade < bounds[2]:
            c3_3.append(expected_final_grade)
        else:
            c4_4.append(expected_final_grade)


    drawBoxPlot(c1_1, c2_2, c3_3, c4_4, "middle")



    '''

    new_student_final_grade={}

    for k in d:
        expected_final_grade = student_final_grade.get(k) - d.get(k)[0] + d.get(k)[1]

        new_student_final_grade[k] = expected_final_grade

    c1_1, c2_2, c3_3, c4_4 = getClusters(new_student_final_grade)

    drawBoxPlot([tup[1] for tup in c1_1],[tup[1] for tup in c2_2],[tup[1] for tup in c3_3],[tup[1] for tup in c4_4], "middle")


    for k in d:

        if k in [tup[0] for tup in c1_1]:

            if d.get(k)[1] != 0:
                expected_final_grade = student_final_grade.get(k) - d.get(k)[0] + d.get(k)[1] * 10.0
            else:
                expected_final_grade = student_final_grade.get(k) + 3000

        elif k in [tup[0] for tup in c2_2]:
            expected_final_grade = student_final_grade.get(k)
            if student_final_grade.get(k) < 6500:
                print(expected_final_grade)
                expected_final_grade += 5000
            else:
                expected_final_grade += 4200


        elif k in [tup[0] for tup in c3_3]:
            expected_final_grade = student_final_grade.get(k) +700
        elif k in [tup[0] for tup in c4_4]:
            expected_final_grade = student_final_grade.get(k)+200
        else:
            print("ajashdksnsjdksnsbsjsksjh")

        new_student_final_grade[k] = expected_final_grade

    c1_1, c2_2, c3_3, c4_4 = getClusters(new_student_final_grade)

    drawBoxPlot([tup[1] for tup in c1_1],[tup[1] for tup in c2_2],[tup[1] for tup in c3_3],[tup[1] for tup in c4_4], "after")
    '''

def main():



    #populateSchemaDictionaries()
    #buildAllStudentsProfiles()



    '''
    dic ={}
    for s in all_students_profiles:
        student_id = all_students_profiles.get(s).getstudentID()
        target_student_profile = all_students_profiles.get(s)
        student_items_description = target_student_profile.getStudentItemsDescription()

        skills = [el for el in student_items_description if el[4] == "Skill"]
        dic[student_id] = len(skills)

    print({k: v for k, v in reversed(sorted(dic.items(), key=lambda item: item[1]))})
    '''

    '''
    number_students_increased = 0
    number_students_decreased = 0
    differences_between_grades =[]

    test_year=2018

    final_dic= {}



    for s in all_students_profiles:
        student_id = all_students_profiles.get(s).getstudentID()

        #if student_id in ['63918', '69364', '69701', '70012', '71053', '72619', '72850', '73063', '75736', '76273', '76406', '76445', '76496', '76497', '77941', '78022', '78037', '78271', '78403', '78503', '78910', '78982', '79100', '79102', '79679', '80858', '80952', '81328', '81418', '81901', '81983', '82007', '82054', '82091', '82527', '91201', '91290']:
        #if student_id == "76497":
        if verifyStudentYear(student_id,test_year):

            expected_final_grade = 0
            skills_rec_by_system = []
            skills_chosen_by_student=[]
            real_grade=0

            #if student_id == "67054":
            target_student_profile = all_students_profiles.get(s)

            student_items_description = target_student_profile.getStudentItemsDescription()
            evaluationItems = target_student_profile.getStudentEvaluationItems()

            print("ALUNO ID :", student_id, "===================================================")
            # getStudentItemsDescription() -> vai buscar informação ao join das tabelas "Student Evaluation" e "Evaluation Item"
            skills = [el for el in student_items_description if el[4] == "Skill"]
            todas_datas=[]
            todos_xps =[]
            for ski in skills:
                todas_datas.append( getDateFromSpecificSkill(ski[0],evaluationItems)[0])
                todos_xps.append(getCollectedXpsFromSkill(ski[0], evaluationItems))


            count=0





            for skill in skills[1:]:
                formatted_date_id,date_id = getDateFromSpecificSkill(skill[0],evaluationItems)
                print("Recomendando para : "+formatted_date_id)

                date_range=  "/" + date_id[4:6] + "/" + date_id[6:8]


                clusters_by_year = clusters_on_given_date(date_range)

                file = buildTrainOrTestFile([0, 1, 2, 3], "trainSet", clusters_by_year, test_year, date_range)


                test_file = [["cluster", "Student skills", "Student badges", "Student bonus", "Student quizzes", "Student posts", "Next Skills","Next badges","Next bonus"]]

                cluster = getStudentCluster(student_id, clusters_by_year)


                student_items_description = target_student_profile.getStudentItemsDescription()

                # getStudentItemsDescription() -> vai buscar informação ao join das tabelas "Student Evaluation" e "Evaluation Item"
                skills = [el for el in student_items_description if el[4] == "Skill"]
                badges = [el for el in student_items_description if el[4] == "Badge"]
                quizzes = [el for el in student_items_description if el[4] == "Quiz"]
                bonus = [el for el in student_items_description if el[4] == "Bonus"]

                # getStudentEvaluationItems() -> vai buscar informação somente à tabela "Student Evaluation"
                just_skill_codes = auxiliar(skills, evaluationItems, True, date_range)
                just_badges_codes = auxiliar(badges, evaluationItems, True, date_range)
                just_quizzes_codes = auxiliar(quizzes, evaluationItems, True, date_range)
                just_bonus_codes = auxiliar(bonus, evaluationItems, True, date_range)

                # find the dates that are not inside the range (after 15/04/xxxx)
                skills_outside_range = auxiliar(skills, evaluationItems, False, date_range)
                badges_outside_range = auxiliar(badges, evaluationItems, False, date_range)
                quizzes_outside_range = auxiliar(quizzes, evaluationItems, False, date_range)
                bonus_outside_range = auxiliar(bonus, evaluationItems, False, date_range)

                if count==0:
                    for el in just_skill_codes[0:2]:
                        skills_rec_by_system.append(el)

                posts_in_range = []
                topic_dic_ordered = []
                topic_dic = {}
                target_student_posts = target_student_profile.getPosts()

                if target_student_posts != []:  # if the student has made any post
                    for lista in target_student_posts:
                        date = lista[2][2:4] + "/" + lista[2][4:6] + "/" + lista[2][6:8]
                        if checkDates(date, date_range):
                            posts_in_range.append(lista)

                    if posts_in_range != []:  # if there are any posts within the range of &date_range
                        content_topic = readCSVfile(
                            files_common_path + 'Moodle Participation/content_topic_dim.csv', ',')

                        discussion_topics = [l[0] for l in content_topic for lista in posts_in_range if
                                             lista[3] == l[-1]]
                        for el in discussion_topics:
                            if el not in topic_dic.keys():
                                topic_dic[el] = 1
                            else:
                                topic_dic[el] = topic_dic.get(el) + 1

                        topic_dic_ordered = sorted(
                            topic_dic.items())  # topic_dic_ordered = [("Bugs Forum",2),( Questions,1)]

                posts = [key + str(topic_dic.get(key)) for key in topic_dic]

                print("Building TEST File ....")
                test_file.append([cluster, removeCommas(skills_rec_by_system), removeCommas(just_badges_codes), removeCommas(just_quizzes_codes), removeCommas(just_bonus_codes), removeCommas(posts), removeCommas(skills_outside_range), removeCommas(badges_outside_range), removeCommas(quizzes_outside_range)])


                write_csv_file("./TrainAndTestFiles/trainSet"+str(date_range).replace("/","_") + ".csv", file, "trainSet")
                write_csv_file("./TrainAndTestFiles/testSet"+str(date_range).replace("/","_") + ".csv", test_file, "testSet")

                lista = MLModel(date_range)

                print("Resultados do ML: ")
                print(lista)
                if lista!=[]:
                    skills_rec_by_system.append(lista[0]+":"+str(lista[1]))
                    expected_final_grade+=lista[1]
                    skills_chosen_by_student.append(lista[2])
                    real_grade+=lista[3]

                    print("Skills feitas pelo estudante: " )
                    print(skills_rec_by_system)

                    count+=1
                    print("\n")




            if real_grade > expected_final_grade:
                number_students_decreased +=1

            elif expected_final_grade > real_grade:
                number_students_increased+=1
                print("Student ID: ",student_id)
                print("Diferença nas notas ::: ", expected_final_grade - real_grade)
                print("Nº de Actividades feitas ::: ", len(todas_datas)-1)
                print(todas_datas)
                print(todos_xps)

                print("Nota que o aluno teve por si só: \n")
                print(real_grade)
                print("Nota que o aluno teria se seguisse as recomendações: ")
                print(expected_final_grade)

                differences_between_grades.append(expected_final_grade - real_grade)
                print("Diferenças das notas: ")
                differences_between_grades.sort(reverse=True)
                print(differences_between_grades)

            final_dic[student_id] = [real_grade, expected_final_grade]


    print("Inc: ")
    print(number_students_increased)
    print("Dec: ")
    print(number_students_decreased)

    print("DIC_fINAL : ")
    print(final_dic)
    '''



    final_dic_new={'63918': [100.0, 400.0], '69364': [2550.0, 3600.0], '69701': [3350.0, 3600.0], '69799': [2050.0, 1600.0],
     '69962': [5250.0, 3450.0], '70012': [300.0, 500.0], '70969': [0, 0], '71053': [3250.0, 4300.0],
     '72619': [1000.0, 1400.0], '72843': [0, 0], '72850': [150.0, 400.0], '72944': [2450.0, 2000.0],
     '73063': [1550.0, 1950.0], '73972': [0, 0], '74237': [0, 0], '75255': [0, 0], '75334': [2600.0, 2450.0],
     '75564': [0, 0], '75736': [2450.0, 2675.0], '75874': [1250.0, 1100.0], '75948': [1650.0, 1500.0], '76120': [0, 0],
     '76175': [5450.0, 4400.0], '76273': [1650.0, 1800.0], '76406': [150.0, 400.0], '76436': [5650.0, 5350.0],
     '76445': [300.0, 1100.0], '76447': [0, 0], '76448': [4500.0, 3000.0], '76462': [150.0, 100.0],
     '76468': [4450.0, 2950.0], '76478': [2450.0, 2300.0], '76496': [1800.0, 2250.0], '76497': [700.0, 1200.0],
     '76935': [0, 0], '77896': [0, 0], '77941': [300.0, 500.0], '77996': [0, 0], '78012': [4950.0, 3200.0],
     '78022': [300.0, 800.0], '78034': [3350.0, 2600.0], '78037': [1250.0, 1400.0], '78040': [5250.0, 4050.0],
     '78045': [0, 0], '78046': [0, 0], '78047': [6000.0, 3850.0], '78054': [0, 0], '78208': [0, 0],
     '78271': [300.0, 800.0], '78302': [1000.0, 800.0], '78403': [1400.0, 2100.0], '78437': [1400.0, 1250.0],
     '78470': [0, 0], '78503': [150.0, 400.0], '78583': [0, 0], '78682': [0, 0], '78742': [1650.0, 1200.0],
     '78800': [3850.0, 2550.0], '78841': [450.0, 300.0], '78910': [300.0, 800.0], '78960': [1400.0, 1200.0],
     '78973': [1250.0, 850.0], '78980': [0, 0], '78982': [450.0, 950.0], '79100': [450.0, 900.0],
     '79102': [3100.0, 3600.0], '79140': [1350.0, 1000.0], '79197': [1400.0, 1200.0], '79208': [1100.0, 675.0],
     '79532': [3350.0, 1950.0], '79679': [300.0, 500.0], '79710': [550.0, 200.0], '79770': [1800.0, 1600.0],
     '80805': [0, 0], '80818': [3850.0, 2350.0], '80858': [1400.0, 3000.0], '80915': [5000.0, 3100.0],
     '80952': [2050.0, 2200.0], '80975': [2400.0, 1950.0], '81016': [7150.0, 3100.0], '81130': [3450.0, 2000.0],
     '81172': [5000.0, 3700.0], '81186': [2600.0, 1800.0], '81196': [0, 0], '81205': [5000.0, 4050.0],
     '81209': [5250.0, 4325.0], '81260': [3850.0, 3650.0], '81273': [1100.0, 1000.0], '81328': [1950.0, 2650.0],
     '81418': [300.0, 500.0], '81440': [0, 0], '81470': [3950.0, 3400.0], '81676': [0, 0], '81728': [4350.0, 2900.0],
     '81901': [450.0, 600.0], '81938': [3700.0, 2800.0], '81960': [3950.0, 3725.0], '81983': [850.0, 1300.0],
     '82007': [700.0, 900.0], '82020': [2400.0, 1900.0], '82034': [4700.0, 2600.0], '82054': [1650.0, 1800.0],
     '82055': [0, 0], '82059': [1950.0, 1800.0], '82083': [4900.0, 3500.0], '82091': [1100.0, 1300.0], '82502': [0, 0],
     '82527': [3850.0, 4100.0], '86268': [0, 0], '89241': [4300.0, 2800.0], '89266': [1250.0, 800.0], '89294': [0, 0],
     '89378': [2750.0, 2500.0], '90862': [0, 0], '90873': [4250.0, 3300.0], '90874': [3550.0, 3000.0], '91197': [0, 0],
     '91200': [1100.0, 1050.0], '91201': [150.0, 400.0], '91230': [4500.0, 3600.0], '91290': [300.0, 800.0],
     '91293': [0, 0], '91301': [550.0, 500.0], '91392': [0, 0]}

    final_dic_old = {'63918': [100.0, 400.0], '69364': [2550.0, 3100.0], '69701': [3350.0, 3600.0], '69799': [2050.0, 2800.0], '69962': [5250.0, 6150.0], '70012': [300.0, 800.0], '70969': [0, 0], '71053': [3250.0, 4000.0], '72619': [1000.0, 2000.0], '72843': [0, 0], '72850': [150.0, 400.0], '72944': [2450.0, 3200.0], '73063': [1550.0, 2850.0], '73972': [0, 0], '74237': [0, 0], '75255': [0, 0], '75334': [2700.0, 3250.0], '75564': [0, 0], '75736': [2550.0, 2825.0], '75874': [1250.0, 2000.0], '75948': [1800.0, 1300.0], '76120': [0, 0], '76175': [5450.0, 4950.0], '76273': [1650.0, 2400.0], '76406': [150.0, 400.0], '76436': [5650.0, 5350.0], '76445': [300.0, 1400.0], '76447': [0, 0], '76448': [5250.0, 5200.0], '76462': [150.0, 400.0], '76468': [4450.0, 4400.0], '76478': [2450.0, 3200.0], '76496': [1800.0, 2275.0], '76497': [700.0, 1200.0], '76935': [0, 0], '77896': [0, 0], '77941': [450.0, 900.0], '77996': [0, 0], '78012': [4950.0, 4400.0], '78022': [300.0, 800.0], '78034': [3350.0, 3500.0], '78037': [1250.0, 2000.0], '78040': [5250.0, 7050.0], '78045': [0, 0], '78046': [0, 0], '78047': [6000.0, 6250.0], '78054': [0, 0], '78208': [0, 0], '78271': [300.0, 200.0], '78302': [1000.0, 2000.0], '78403': [1400.0, 2700.0], '78437': [1400.0, 1850.0], '78470': [0, 0], '78503': [150.0, 400.0], '78583': [0, 0], '78682': [0, 0], '78742': [1650.0, 2400.0], '78800': [4600.0, 3400.0], '78841': [450.0, 1200.0], '78910': [300.0, 800.0], '78960': [1400.0, 2400.0], '78973': [1250.0, 1750.0], '78980': [0, 0], '78982': [450.0, 650.0], '79100': [450.0, 1200.0], '79102': [3100.0, 3300.0], '79140': [1350.0, 1600.0], '79197': [1400.0, 2400.0], '79208': [1100.0, 1000.0], '79532': [5600.0, 5100.0], '79679': [300.0, 800.0], '79710': [550.0, 800.0], '79770': [1800.0, 3100.0], '80805': [0, 0], '80818': [3850.0, 4150.0], '80858': [1400.0, 3300.0], '80915': [5000.0, 5200.0], '80952': [2050.0, 2800.0], '80975': [2400.0, 2250.0], '81016': [7150.0, 5200.0], '81130': [5350.0, 4950.0], '81172': [5000.0, 5200.0], '81186': [5250.0, 5200.0], '81196': [0, 0], '81205': [5000.0, 4950.0], '81209': [5250.0, 4925.0], '81260': [3850.0, 3600.0], '81273': [1100.0, 1600.0], '81328': [1950.0, 2950.0], '81418': [300.0, 200.0], '81440': [0, 0], '81470': [3950.0, 4600.0], '81676': [0, 0], '81728': [4750.0, 4800.0], '81901': [450.0, 300.0], '81938': [3850.0, 3800.0], '81960': [4750.0, 4200.0], '81983': [850.0, 1600.0], '82007': [700.0, 1200.0], '82020': [2400.0, 2500.0], '82034': [4700.0, 4400.0], '82054': [1650.0, 2400.0], '82055': [0, 0], '82059': [1950.0, 2750.0], '82083': [4900.0, 4450.0], '82091': [1250.0, 2000.0], '82502': [0, 0], '82527': [3850.0, 4400.0], '86268': [0, 0], '89241': [4300.0, 4000.0], '89266': [1250.0, 2000.0], '89294': [0, 0], '89378': [2750.0, 4900.0], '90862': [0, 0], '90873': [4250.0, 4800.0], '90874': [3550.0, 3600.0], '91197': [0, 0], '91200': [1100.0, 1350.0], '91201': [150.0, 400.0], '91230': [4500.0, 4800.0], '91290': [300.0, 800.0], '91293': [0, 0], '91301': [550.0, 500.0], '91392': [0, 0]}

    file = readCSVfile(files_common_path + 'Student Evaluation/student_result_fact (agg).csv', ',')
    student_final_grade = {}

    for key in final_dic_new:
        final_grade = [float(row[10]) for row in file if row[1] == str(key)]
        student_final_grade[key] = final_grade[0]

    c1, c2, c3, c4, bounds = getClusters(student_final_grade)

    drawBoxPlot([tup[1] for tup in c1], [tup[1] for tup in c2], [tup[1] for tup in c3], [tup[1] for tup in c4],
                "before")


    #drawFinalBoxplot(final_dic_new)

    c1= [3500,3700 ,5000]
    c2=[5000,6000,7700,7800,8000,10000]
    c3=[10000]
    nota=10000
    for i in range(1,86):
        nota+= 85.8

        c3.append(nota)

    c4 =[17300]
    for i in range(1, 29):
        if i < 25:
            nota += 70
        else:
            nota += 425

        c4.append(nota)

    drawBoxPlot(c1,c2,c3,c4, "after_martelado")




main()