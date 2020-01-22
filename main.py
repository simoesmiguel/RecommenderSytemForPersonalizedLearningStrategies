import matplotlib.pyplot as plt
import csv
from functools import reduce
import plotly.figure_factory as ff
import plotly.graph_objects as go
from collections import Counter

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

def getEvaluationItems(all_clusters, tag):
    eiUnderachievers, eiHalfhearted, eiRegular, eiAchievers = [], [], [], []  # evaluation items

    for el in all_clusters:
        for i in range(len(el)):
            student_id = el[i][0]  # studentID
            student_profile_id = all_students_profiles[student_id].getStudentItemsDescription() # returns the student Profile items Description
            if(student_profile_id!=[]):
                new_list=[]
                if tag == "skill": new_list = [el for el in student_profile_id if (el[2] == "Skill")]

                if all_clusters.index(el) == 0: eiUnderachievers += [row[1] for row in new_list]
                elif all_clusters.index(el) == 1: eiHalfhearted += [row[1] for row in new_list]
                elif all_clusters.index(el) == 2: eiRegular += [row[1] for row in new_list]
                else: eiAchievers += [row[1] for row in new_list]

    return  eiUnderachievers, eiHalfhearted, eiRegular, eiAchievers


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
        drawHistogram([el for el in l], "evaluationItems", "2")




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

    elif flag == "evaluationItems" and type == "2":
        colors = ['red', 'tan', 'lime','green']
        labels = ["UnderAchievers", "HalfHearted", "Regular", "Achievers"]

        fig, axes = plt.subplots(nrows=2, ncols=2)
        ax0, ax1, ax2, ax3 = axes.flatten()

        x_dic = Counter(l[0])
        x = [x_dic[el] for el in l[0]]

        ax0.barh(l[0], x, align='center', color=colors[0])
        ax0.set_yticks(l[0])
        ax0.set_yticklabels(l[0])
        ax0.invert_yaxis()  # labels read top-to-bottom
        ax0.set_xlabel('Number of Students')
        ax0.set_title('Underachievers skills')

        y=l[1]
        x_dic = Counter(y)
        x = [x_dic[el] for el in y]

        ax1.barh(y, x, align='center', color=colors[0])
        ax1.set_yticks(y)
        ax1.set_yticklabels(y)
        ax1.invert_yaxis()  # labels read top-to-bottom
        ax1.set_xlabel('Number of Students')
        ax1.set_title('HalfHearted skills')

        y = l[2]
        x_dic = Counter(y)
        x = [x_dic[el] for el in y]

        ax2.barh(y, x, align='center', color=colors[0])
        ax2.set_yticks(y)
        ax2.set_yticklabels(y)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_xlabel('Number of Students')
        ax2.set_title('Regular skills')

        y = l[3]
        x_dic = Counter(y)
        x = [x_dic[el] for el in y]

        ax3.barh(y, x, align='center', color=colors[0])
        ax3.set_yticks(y)
        ax3.set_yticklabels(y)
        ax3.invert_yaxis()  # labels read top-to-bottom
        ax3.set_xlabel('Number of Students')
        ax3.set_title('Achievers skills')

        fig.tight_layout()
        plt.show()



        '''
        fig, axes = plt.subplots(nrows=2, ncols=2)
        ax0, ax1, ax2, ax3 = axes.flatten()

        ax0.hist(l[0], density=True, histtype='bar', color=colors[0], label=labels[0])
        ax0.legend(prop={'size': 10})

        ax1.hist(l[1], density=True, histtype='bar', color=colors[1], label=labels[1])
        ax0.legend(prop={'size': 10})

        ax2.hist(l[2], density=True, histtype='bar', color=colors[2], label=labels[2])
        ax0.legend(prop={'size': 10})

        ax3.hist(l[3], density=True, histtype='bar', color=colors[3], label=labels[3])
        ax0.legend(prop={'size': 10})

        fig.tight_layout()
        plt.show()
        '''






def main():
    populateSchemaDictionaries()
    underachievers, halfhearted, regular, achievers = clusters()
    # drawplot([underachievers, halfhearted, regular, achievers])

    # print(len([float(row[10]) for row in  SE["student_result_fact"] if float(row[10]) > 20000]))
    buildAllStudentsProfiles()

    statisticalAnalysis(underachievers, halfhearted, regular, achievers, "evaluationItems_skill")


if __name__ == "__main__":
    main()
