import matplotlib.pyplot as plt
import csv
from functools import reduce
import plotly.figure_factory as ff
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime as dt


from sklearn.cluster import KMeans

import numpy as np

files_common_path = '/Users/miguelsimoes/Documents/Tese/Final Data Warehouse/'

SE = {}  # Student Evaluation Schema
MP = {}  # Moodle Participation
all_students_profiles={} # saves all the instances of students profiles

class Student:
    def __init__(self, studentID):
        self.studentID = studentID

    studentResults = []
    studentEvaluationItems = []
    studentItemsDescription =[]
    moodleParticipation = []
    posts=[]
    messageAnalysis=[]

    #setters
    def setStudentResults(self, sr):
        self.studentResults = sr

    def setStudentEvaluationItems(self, se):
        self.studentEvaluationItems = se

    def setStudentItemsDescription(self, sid):
        self.studentItemsDescription = sid

    def setMoodleParticipation(self, mp):
        self.moodleParticipation=mp

    def setPosts(self,p):
        self.posts=p

    def setMessageAnalysis(self,ma):
        self.messageAnalysis=ma

    #getters
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
def readCSVfile(csv_file_name):
    l = []
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0: line_count += 1
            else:
                l.append(row)
                line_count += 1
        return l


def populateSchemaDictionaries():
    SE["student_result_fact"] = readCSVfile(files_common_path + 'Student Evaluation/student_result_fact (agg).csv')
    SE["student_evaluation_fact"] = readCSVfile(files_common_path + 'Student Evaluation/student_evaluation_fact.csv')
    SE["student_dim"] = readCSVfile(files_common_path + 'Student Evaluation/student_dim.csv')
    SE["semester_dim"] = readCSVfile(files_common_path + 'Student Evaluation/semester_dim.csv')
    SE["evaluation_item_dim"] = readCSVfile(files_common_path + 'Student Evaluation/evaluation_item_dim.csv')
    SE["date_dim"] = readCSVfile(files_common_path + 'Student Evaluation/date_dim.csv')

    MP["action_dim"] = readCSVfile(files_common_path + 'Moodle Participation/action_dim.csv')
    MP["content_topic_dim"] = readCSVfile(files_common_path + 'Moodle Participation/content_topic_dim.csv')
    MP["date_dim"] = readCSVfile(files_common_path + 'Moodle Participation/date_dim.csv')
    MP["logs_fact"] = readCSVfile(files_common_path + 'Moodle Participation/logs_fact.csv')
    MP["message_analysis_fact"] = readCSVfile(
        files_common_path + 'Moodle Participation/message_analysis_fact (agg).csv')
    MP["message_dim"] = readCSVfile(files_common_path + 'Moodle Participation/message_dim.csv')
    MP["moodle_participation_fact"] = readCSVfile(
        files_common_path + 'Moodle Participation/moodle_participation_fact (agg).csv')
    MP["posts_fact"] = readCSVfile(files_common_path + 'Moodle Participation/posts_fact.csv')
    MP["semester_dim"] = readCSVfile(files_common_path + 'Moodle Participation/semester_dim.csv')
    MP["student_dim"] = readCSVfile(files_common_path + 'Moodle Participation/student_dim.csv')
    MP["web_element_dim"] = readCSVfile(files_common_path + 'Moodle Participation/web_element_dim.csv')


def clusters():
    all_scores = [float(row[10]) for row in SE["student_result_fact"]]
    highest_score = max(all_scores)
    c1 = [(float(row[1]), float(row[10])) for row in SE["student_result_fact"] if
          float(row[10]) <= 0.25 * highest_score]
    c2 = [(float(row[1]), float(row[10])) for row in SE["student_result_fact"] if
          0.25 * highest_score < float(row[10]) <= 0.50 * highest_score]
    c3 = [(float(row[1]), float(row[10])) for row in SE["student_result_fact"] if
          0.50 * highest_score < float(row[10]) <= 0.75 * highest_score]
    c4 = [(float(row[1]), float(row[10])) for row in SE["student_result_fact"] if float(row[10]) > 0.75 * highest_score]

    return c1, c2, c3, c4


def sortSecond(val):
    return val[1]


def drawplot(clusters):  # plots the

    list1 = clusters[0]
    list1.sort(key=sortSecond)
    x = [tup[0] for tup in list1]
    y = [tup[1] for tup in list1]
    plt.scatter(x, y, c='coral')

    list2 = clusters[1]
    list2.sort(key=sortSecond)
    x = [tup[0] for tup in list2]
    y = [tup[1] for tup in list2]
    plt.scatter(x, y, c='blue')

    list3 = clusters[2]
    list3.sort(key=sortSecond)
    x = [tup[0] for tup in list3]
    y = [tup[1] for tup in list3]
    plt.scatter(x, y, c='red')

    list4 = clusters[3]
    list4.sort(key=sortSecond)
    x = [tup[0] for tup in list4]
    y = [tup[1] for tup in list4]
    plt.scatter(x, y, c='green')

    plt.title('Scatter plot')
    plt.xlabel('studentID')
    plt.ylabel('final XP')
    plt.show()


def buildProfile(studentId):
    studentProfile = Student(studentId)
    studentProfile.setStudentResults([row for row in SE["student_result_fact"] if row[1] == str(studentId)])
    studentProfile.setStudentEvaluationItems([row for row in SE["student_evaluation_fact"] if row[1] == str(studentId)])
    studentProfile.setStudentItemsDescription([ row1 for row in SE["student_evaluation_fact"] if row[1] == str(studentId)
                                                for row1 in SE["evaluation_item_dim"] if row[2] == row1[0]])
    studentProfile.setMoodleParticipation([row for row in MP["moodle_participation_fact"] if row[1] == str(studentId)])
    studentProfile.setPosts([row for row in MP["posts_fact"] if row[1] == str(studentId)])
    studentProfile.setMessageAnalysis([row for row in MP["message_analysis_fact"] if row[1] == str(studentId)])

    all_students_profiles[float(studentId)] = studentProfile # save the student Profile

def buildAllStudentsProfiles():
    count=0
    for i in range(len(SE["student_result_fact"])):
        buildProfile(SE["student_result_fact"][i][1])
        count+=1
    print("Created ",count, " Student Profiles")

def getTotalPostsForAllClusters(all_clusters):
    tpUnderachievers, tpHalfhearted, tpRegular, tpAchievers = [],[],[],[]  # total posts underachievers

    for el in all_clusters:
        for i in range(len(el)):
            student_id = el[i][0]  # studentID
            student_profile_mp = all_students_profiles[student_id].getMoodleParticipation() # returns the student Profile moodle Participation
            if (student_profile_mp != []):  # there are students who don't have any posts in the forum
                if all_clusters.index(el)==0 : tpUnderachievers.append(float(student_profile_mp[0][10]))
                elif all_clusters.index(el)==1 : tpHalfhearted.append(float(student_profile_mp[0][10]))
                elif all_clusters.index(el)==2 : tpRegular.append(float(student_profile_mp[0][10]))
                else: tpAchievers.append(float(student_profile_mp[0][10]))

    return tpUnderachievers, tpHalfhearted, tpRegular, tpAchievers


def getTotalXPAllClusters(all_clusters, tag):
    xpUnderachievers, xpHalfhearted, xpRegular, xpAchievers = [],[],[],[]  # total posts

    index=0
    if(tag=="xpFromQuizzes"): index = 8
    elif(tag=="xpFromSkillTree"): index = 9

    for el in all_clusters:
        for i in range(len(el)):
            student_id = el[i][0]  # studentID
            student_profile_results = all_students_profiles[student_id].getStudentResults()
            if all_clusters.index(el) == 0: xpUnderachievers.append(float(student_profile_results[0][index]))
            elif all_clusters.index(el) == 1: xpHalfhearted.append(float(student_profile_results[0][index]))
            elif all_clusters.index(el) == 2: xpRegular.append(float(student_profile_results[0][index]))
            else: xpAchievers.append(float(student_profile_results[0][index]))

    return xpUnderachievers, xpHalfhearted, xpRegular, xpAchievers

def remove_duplicates(x):
  return list(dict.fromkeys(x))

def getEvaluationItems(all_clusters, tag):
    eiUnderachievers, eiHalfhearted, eiRegular, eiAchievers = [], [], [], []  # evaluation items

    for el in all_clusters:
        for i in range(len(el)):
            in_range=False
            student_id = el[i][0]  # studentID
            student_profile_id = all_students_profiles[student_id].getStudentItemsDescription() # returns the student Profile items Description

            evaluationItems=all_students_profiles[student_id].getStudentEvaluationItems()

            if student_profile_id:

                new_list = [] #esta lista vai ter só os items que nos interessam (skill ou Badge)
                if tag == "skill": new_list = [el for el in student_profile_id if (el[2] == "Skill")]
                elif tag == "badge": new_list = [el for el in student_profile_id if (el[2] == "Badge")]


                #ir buscar a data de cada item e ver se foi efetuada no primeiro semestre
                already_found=False
                for el1 in new_list:
                    item_id = el1[0]
                    for el2 in evaluationItems:
                        if item_id == el2[2]: # se os item_id coincidirem
                            already_found=True
                            date_id = el2[0]
                            formatted_date_id = date_id[2:4]+"/"+date_id[4:6]+"/"+date_id[6:8]
                            if checkDates(formatted_date_id): # checkar se a data está na primeira metade do semestre
                                in_range = True
                            break

                    if already_found: break


                if in_range:
                    this_student_badge_list=[row[1] for row in new_list]
                    badge_ntimes = Counter(this_student_badge_list) # given this students' badges, this dictionary saves the number of times that the student has each badge
                    this_student_badge_list = remove_duplicates(this_student_badge_list)

                    if all_clusters.index(el) == 0: eiUnderachievers += this_student_badge_list
                    elif all_clusters.index(el) == 1: eiHalfhearted += this_student_badge_list
                    elif all_clusters.index(el) == 2: eiRegular += this_student_badge_list
                    else: eiAchievers += this_student_badge_list



    return  eiUnderachievers, eiHalfhearted, eiRegular, eiAchievers


def checkDates(d):
    all_dates={('2011/02/11', '2011/06/29'): '0', ('2012/02/13', '2012/07/03'): '1', ('2013/02/11', '2013/07/16'): '2',
     ('2014/02/14', '2014/07/01'): '3', ('2015/02/13', '2015/06/24'): '4', ('2016/02/12', '2016/07/02'): '5',
     ('2017/02/17', '2017/06/14'): '6', ('2018/02/14', '2018/09/21'): '7', ('2013/03/24', '2013/06/23'): '2',
     ('2014/02/18', '2014/06/17'): '3', ('2015/02/21', '2015/06/23'): '4', ('2016/02/09', '2016/06/20'): '5',
     ('2017/02/21', '2017/06/06'): '6', ('2018/02/15', '2018/07/05'): '7', ('2019/02/14', '2019/05/09'): '8',
     ('2011/02/14', '2011/06/26'): '0', ('2012/02/14', '2012/06/30'): '1', ('2013/05/04', '2013/07/22'): '2'}

    for el in all_dates:
        d1 = dt.strptime(el[0][2:], "%y/%m/%d")
        d2 = dt.strptime(el[0][2:4]+"/04/15", "%y/%m/%d")
        if(d1 < dt.strptime(d, "%y/%m/%d") < d2): # quer dizer que está na primeira metade do semestre (até 15 de Abril)
            return True
    return False


def statisticalAnalysis(u, h, r, a, tag):

    if tag == "totalPostsMoodle":
        l = getTotalPostsForAllClusters([u, h, r, a])
        drawHistogram([el for el in l], "totalPosts", "2")
    elif tag == "totalXPfromQuizzes":
        l = getTotalXPAllClusters([u, h, r, a], "xpFromQuizzes")
        drawHistogram([el for el in l], "totalXPQuizzes", "2")
    elif tag == "totalXPfromSkillTree":
        l = getTotalXPAllClusters([u, h, r, a], "xpFromSkillTree")
        drawHistogram([el for el in l], "totalXPSkillTree", "2")
    elif tag == "evaluationItems_skill":
        l = getEvaluationItems([u, h, r, a], "skill")
        drawBarchart([el for el in l], tag="skill", ylabel="Activities")
    elif tag == "evaluationItems_badges":
        l = getEvaluationItems([u, h, r, a], "badge")
        drawBarchart([el for el in l], tag="badge", ylabel="Badges",
                     titles=["Underachievers Badges", "HalfHearted Badges", "Regular Students Badges",
                             "Achievers Badges"],
                     rotations=[0,0,0,0]
                     )


#recursive function
def drawBarchart(l, tag , ylabel, colors = ['red', 'tan', 'lime','green'],
                 titles = ["Underachievers Skills", "HalfHearted Skills", "Regular Students Skills", "Achievers Skills"],
                 rotations =[25,25,15,0]):
    if l: # stop condition
        y = l[0]
        x_dic = Counter(y)
        x = [x_dic[el] for el in y]

        plt.figure(figsize=(14, 10))
        plt.barh(y, x, align='center', color=colors[0])
        plt.yticks(y, rotation=rotations[0])
        plt.ylabel(ylabel)
        plt.xlabel('Number of Students')
        plt.title(titles[0])
        plt.show()

        return drawBarchart(l[1:], tag, ylabel, colors[1:], titles[1:],rotations[1:])


def drawHistogram(l, flag, type):

    if flag == "totalPosts" and type == "1":
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=l[0], name="underAchievers"))
        fig.add_trace(go.Histogram(x=l[1], name="HalfHearted"))
        fig.add_trace(go.Histogram(x=l[2], name="Regular"))
        fig.add_trace(go.Histogram(x=l[3], name="Achievers"))

        # The two histograms are drawn on top of another
        fig.update_layout(barmode='stack',
                          title_text='Number of Posts by Profile',  # title of plot
                          xaxis_title_text='Nº Posts',  # xaxis label
                          yaxis_title_text='Count',  # yaxis label
                          )
        fig.show()

    if flag == "totalPosts" and type == "2":
        colors = ['red', 'tan', 'lime','green']
        plt.hist(l, histtype='bar', color=colors, label=["UnderAchievers", "HalfHearted", "Regular", "Achievers"], stacked=True)
        plt.legend(prop={'size': 10})
        plt.title('Number of Posts by Profile')
        plt.xlabel('Nº of posts')
        plt.ylabel('Counts')
        plt.show()

    elif flag == "totalXPQuizzes" and type == "2":
        colors = ['red', 'tan', 'lime','green']
        plt.hist(l, histtype='bar', color=colors, label=["UnderAchievers", "HalfHearted", "Regular", "Achievers"], stacked=False)
        plt.legend(prop={'size': 10})
        plt.title('Total Quizzes XP by Profiles')
        plt.xlabel('XP')
        plt.ylabel('Counts')
        plt.show()

    elif flag == "totalXPSkillTree" and type == "2":
        colors = ['red', 'tan', 'lime','green']
        plt.hist(l, histtype='bar', color=colors, label=["UnderAchievers", "HalfHearted", "Regular", "Achievers"], stacked=True)
        plt.legend(prop={'size': 10})
        plt.title('Total Skill Tree XP by Profiles')
        plt.xlabel('XP')
        plt.ylabel('Counts')
        plt.show()



def main():
    populateSchemaDictionaries()
    underachievers, halfhearted, regular, achievers = clusters()
    # drawplot([underachievers, halfhearted, regular, achievers])

    # print(len([float(row[10]) for row in  SE["student_result_fact"] if float(row[10]) > 20000]))
    buildAllStudentsProfiles()

    statisticalAnalysis(underachievers, halfhearted, regular, achievers, "evaluationItems_skill")


if __name__ == "__main__":
    main()
