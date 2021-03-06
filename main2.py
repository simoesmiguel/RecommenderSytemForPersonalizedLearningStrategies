from functools import reduce

import matplotlib.pyplot as plt
import csv
import collections

from collections import Counter
from datetime import datetime as dt
import clusterComparator

import MyMLModel2

import numpy as np

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

#date_range = "/03/15"
#date_range = "/04/15"
#date_range = "/05/03"
date_range = "/04/30"


test_year = 2018

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


def clusters():  # splits the students according to quartiles and returns the 4 different clusters
    all_scores = [float(row[10]) for row in SE["student_result_fact"]]
    highest_score = max(all_scores)
    c1 = [(row[1], float(row[10])) for row in SE["student_result_fact"] if
          float(row[10]) <= 0.25 * highest_score]
    c2 = [(row[1], float(row[10])) for row in SE["student_result_fact"] if
          0.25 * highest_score < float(row[10]) <= 0.50 * highest_score]
    c3 = [(row[1], float(row[10])) for row in SE["student_result_fact"] if
          0.50 * highest_score < float(row[10]) <= 0.75 * highest_score]
    c4 = [(row[1], float(row[10])) for row in SE["student_result_fact"] if float(row[10]) > 0.75 * highest_score]

    return c1, c2, c3, c4


def clusters_on_given_date():
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
            if checkDates(formatted_date_id):  # checkar se a data está na primeira metade do semestre
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


def getTotalPostsForAllClusters(all_clusters):
    tpUnderachievers, tpHalfhearted, tpRegular, tpAchievers = [], [], [], []  # total posts underachievers

    for el in all_clusters:
        for i in range(len(el)):
            student_id = el[i][0]  # studentID
            student_profile_mp = all_students_profiles[
                student_id].getMoodleParticipation()  # returns the student Profile moodle Participation
            if (student_profile_mp != []):  # there are students who don't have any posts in the forum
                if all_clusters.index(el) == 0:
                    tpUnderachievers.append(float(student_profile_mp[0][10]))
                elif all_clusters.index(el) == 1:
                    tpHalfhearted.append(float(student_profile_mp[0][10]))
                elif all_clusters.index(el) == 2:
                    tpRegular.append(float(student_profile_mp[0][10]))
                else:
                    tpAchievers.append(float(student_profile_mp[0][10]))

    return tpUnderachievers, tpHalfhearted, tpRegular, tpAchievers


def getTotalXPAllClusters(all_clusters, tag):
    xpUnderachievers, xpHalfhearted, xpRegular, xpAchievers = [], [], [], []  # total posts

    index = 0
    if (tag == "xpFromQuizzes"):
        index = 8
    elif (tag == "xpFromSkillTree"):
        index = 9

    for el in all_clusters:
        for i in range(len(el)):
            student_id = el[i][0]  # studentID
            student_profile_results = all_students_profiles[student_id].getStudentResults()
            if all_clusters.index(el) == 0:
                xpUnderachievers.append(float(student_profile_results[0][index]))
            elif all_clusters.index(el) == 1:
                xpHalfhearted.append(float(student_profile_results[0][index]))
            elif all_clusters.index(el) == 2:
                xpRegular.append(float(student_profile_results[0][index]))
            else:
                xpAchievers.append(float(student_profile_results[0][index]))

    return xpUnderachievers, xpHalfhearted, xpRegular, xpAchievers


def remove_duplicates(x):
    return list(dict.fromkeys(x))


def getEvaluationItems(all_clusters, tag):
    eiUnderachievers, eiHalfhearted, eiRegular, eiAchievers = [], [], [], []  # evaluation items

    for el in all_clusters:
        for i in range(len(el)):
            in_range = False
            student_id = el[i][0]  # studentID
            student_profile_id = all_students_profiles[
                student_id].getStudentItemsDescription()  # returns the student Profile items Description

            evaluationItems = all_students_profiles[student_id].getStudentEvaluationItems()

            if student_profile_id:

                new_list = []  # esta lista vai ter só os items que nos interessam (skill ou Badge)
                if tag == "skill":
                    new_list = [el for el in student_profile_id if (el[4] == "Skill")]
                elif tag == "badge":
                    new_list = [el for el in student_profile_id if (el[4] == "Badge")]

                # ir buscar a data de cada item e ver se foi efetuada no primeiro semestre
                already_found = False
                for el1 in new_list:
                    item_id = el1[0]
                    for el2 in evaluationItems:
                        if item_id == el2[2]:  # se os item_id coincidirem
                            already_found = True
                            date_id = el2[0]
                            formatted_date_id = date_id[2:4] + "/" + date_id[4:6] + "/" + date_id[6:8]
                            if checkDates(formatted_date_id):  # checkar se a data está na primeira metade do semestre
                                in_range = True
                            break

                    if already_found: break

                if in_range:
                    this_student_badge_list = [row[2] for row in new_list]
                    badge_ntimes = Counter(
                        this_student_badge_list)  # given this students' badges, this dictionary saves the number of times that the student has each badge
                    this_student_badge_list = remove_duplicates(this_student_badge_list)

                    if all_clusters.index(el) == 0:
                        eiUnderachievers += this_student_badge_list
                    elif all_clusters.index(el) == 1:
                        eiHalfhearted += this_student_badge_list
                    elif all_clusters.index(el) == 2:
                        eiRegular += this_student_badge_list
                    else:
                        eiAchievers += this_student_badge_list

    return eiUnderachievers, eiHalfhearted, eiRegular, eiAchievers


def checkDates(d):
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


def checkDatesOutsideRange(d):
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


def auxiliar(skills, evaluationItems, inRange):
    '''

    :param skills:
    :param evaluationItems:
    :param inRange:
    :return: this method returns the skills that are inside or outside the range, depending on the inRange variable, ..
    the first half of the semester, and return them by the oldest to the newest
    '''

    all_skills_in_range = checkDateAndGetCollectedXP(skills, evaluationItems, inRange)
    lista = [all_skills_in_range[key] for key in all_skills_in_range]

    return lista



def checkDateAndGetCollectedXP(items, evaluationItems, checkInsideRange):

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
                    if checkDates(formatted_date_id):  # checkar se a data está na primeira metade do semestre
                        collected_xp = el2[3] # collected XP by the student on this activity
                        final_dic[date_id] = el1[11] + ":"+collected_xp # el1[11] is just the item's finalCode
                        break
                else:
                    if checkDatesOutsideRange(formatted_date_id):
                        collected_xp = el2[3] # collected XP by the student on this activity
                        final_dic[date_id] = el1[11] + ":"+collected_xp
                        break


    return collections.OrderedDict(
        sorted(final_dic.items()))  # retorna por ordem da data (do mais antigo para o mais recente)



def sortListofTuples(l):
    return (sorted(l, key=lambda x: x[0]))

def orderdicbyValue(dic):
    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1][0])}

def orderdicbyKey(dic):
    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[0])}


def write_file2(lista, fileName):
    l = []
    for tuple in lista:
        # l.append(tuple[0] + ["values: "] + tuple[1])
        l.append(tuple[0] + tuple[1] + ["values: "] + tuple[2])
    write_csv_file(files_common_path + "Student Evaluation/" + fileName + ".csv", l)


def veryfyStudentYear(studentID, tag):
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

def getStudentYear(studentID):
    student_profile = all_students_profiles.get(studentID)
    l = student_profile.getStudentResults()
    return int(float(l[0][0]))

def StudentDistributionPerYear(dic_all_years):
    print(dic_all_years)
    my_circle = plt.Circle((0, 0), 0.7, color='white', autopct="%.1f%%")
    # Give color names
    names = []
    sizes = []
    for key in dic_all_years:
        names.append(key)
        sizes.append(dic_all_years[key])


    plt.pie(sizes, labels=names, colors=['red','green','blue','skyblue','orange','yellow','violet','cyan'])
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.title("Data distribution over the years")
    plt.show()

def smartStudentDistributionPerYear(dic_all_years):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    names = []
    data = []
    for key in dic_all_years:
        names.append(key)
        data.append(dic_all_years[key])

    total_students = sum(data)
    for i in range(len(data)):
        names[i] = str(names[i]) + "( "+str(round((data[i]/total_students)*100,1))+ "%)"


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


def getStudentCluster(studentID,clusters_by_year):
    for year in clusters_by_year:
        cluster = 0
        for lista in clusters_by_year.get(year):
            for tupl in lista:
                if tupl[0] == studentID:
                    return cluster
            cluster+=1

def buildTrainOrTestFile(activities_to_recommend, tag, clusters_by_year):
    print("Building "+tag+ " File ....")
    count=1
    dic_all_years={}

    activities_to_recommend_mapping = {0: "Next Skills", 1: "Next badges", 2: "Next bonus",
                                       3: "Next quizzes", 4: "Next posts"}

    recommendations_file = [
        ["cluster", "Student skills", "Student badges", "Student bonus", "Student quizzes", "Student posts"]]
    for index in activities_to_recommend:
        recommendations_file[0].append(activities_to_recommend_mapping.get(index))

    for key in all_students_profiles:

        print("Another: ", count)
        year = getStudentYear(key)
        #print(year)
        if year in dic_all_years:
            dic_all_years[year] +=1
        else:
            dic_all_years[year] =1

        if veryfyStudentYear(key, tag):  # this student did the course before &test_year

            target_student_profile = all_students_profiles.get(key)
            target_student_posts = target_student_profile.getPosts()

            content_topic = []
            posts_in_range = []
            topic_dic_ordered = []
            topic_dic = {}
            if target_student_posts != []:  # if the student has made any post
                for lista in target_student_posts:
                    date = lista[2][2:4] + "/" + lista[2][4:6] + "/" + lista[2][6:8]
                    if checkDates(date):
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
            just_skill_codes = auxiliar(skills, evaluationItems, True)
            just_badges_codes = auxiliar(badges, evaluationItems, True)
            just_quizzes_codes = auxiliar(quizzes, evaluationItems, True)
            just_bonus_codes = auxiliar(bonus, evaluationItems, True)


            #find the dates that are not inside the range (after 15/04/xxxx)
            skills_outside_range = auxiliar(skills, evaluationItems, False)
            badges_outside_range = auxiliar(badges, evaluationItems, False)
            quizzes_outside_range = auxiliar(quizzes, evaluationItems, False)
            bonus_outside_range = auxiliar(bonus, evaluationItems, False)


            list_all_recommendations = [skills_outside_range, badges_outside_range, quizzes_outside_range, bonus_outside_range]

            posts = [key + str(topic_dic.get(key)) for key in topic_dic]

            cluster = getStudentCluster(key, clusters_by_year)

            to_recommend = [cluster, just_skill_codes, just_badges_codes, just_quizzes_codes, just_bonus_codes, posts]

            for index in activities_to_recommend:
                to_recommend.append(list_all_recommendations[index])

            recommendations_file.append(to_recommend)

        count += 1

    #StudentDistributionPerYear(dic_all_years)
    #smartStudentDistributionPerYear(dic_all_years)

    return recommendations_file




def main():

    ''''''

    #    Create Train and Test Set #############################################################################

    '''
    #fileType = "trainSet"
    fileType = "testSet"

    populateSchemaDictionaries()
    buildAllStudentsProfiles()
    clusters_by_year = clusters_on_given_date()

    file = buildTrainOrTestFile([0, 1, 2, 3], fileType, clusters_by_year)

    data = date_range[1:].split("/")
    print("./train&test_files/" + data[1] + "_" + data[0] + "/" + fileType + "_" + str(test_year) + "_dateRange=" +
          data[1] + "_" + data[0] + ".csv")
    write_csv_file(
        "./train&test_files/" + data[1] + "_" + data[0] + "/" + fileType + "_" + str(test_year) + "_dateRange=" + data[
            1] + "_" + data[0] + ".csv", file, fileType)

    print("THAT'S ALL FOLKS")
    '''

    MyMLModel2.main()



    '''
    populateSchemaDictionaries()
    buildAllStudentsProfiles()
    clusters_by_year = clusters_on_given_date()
    total =0
    for year in clusters_by_year:
        print(year)
        print("     U = ",len(clusters_by_year.get(year)[0]))
        print("     H = ",len(clusters_by_year.get(year)[1]))
        print("     R = ",len(clusters_by_year.get(year)[2]))
        print("     A = ",len(clusters_by_year.get(year)[3]))
        total+=len(clusters_by_year.get(year)[0])+len(clusters_by_year.get(year)[1])+len(clusters_by_year.get(year)[2])+len(clusters_by_year.get(year)[3])

    print(total)
    '''

if __name__ == "__main__":
    main()
