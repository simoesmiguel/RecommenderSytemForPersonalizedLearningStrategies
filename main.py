import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans
import numpy as np

files_common_path = '/Users/miguelsimoes/Documents/Tese/Final Data Warehouse/'

SE = {}  # Student Evaluation Schema
MP = {}  # Moodle Participation


class Student:
    def __init__(self, studentID):
        self.studentID = studentID

    studentResults = []
    studentEvaluationItems = []
    moodleParticipation = []
    posts=[]
    messageAnalysis=[]

    #setters
    def setStudentResults(self, sr):
        self.studentResults = sr

    def setStudentEvaluationItems(self, se):
        self.studentEvaluationItems = se

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
            if line_count == 0:
                line_count += 1
            else:
                # print(row)
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
    studentProfile.setMoodleParticipation([row for row in MP["moodle_participation_fact"] if row[1] == str(studentId)])
    studentProfile.setPosts([row for row in MP["posts_fact"] if row[1] == str(studentId)])
    studentProfile.setMessageAnalysis([row for row in MP["message_analysis_fact"] if row[1] == str(studentId)])

def buildAllStudentsProfiles():
    count=0
    for i in range(len(SE["student_result_fact"])):
        buildProfile(SE["student_result_fact"][i][1])
        count+=1
    print("Created ",count, " Student Profiles")

def main():
    populateSchemaDictionaries()
    underachievers, halfhearted, regular, achievers = clusters()
    # drawplot([underachievers, halfhearted, regular, achievers])

    # print(len([float(row[10]) for row in  SE["student_result_fact"] if float(row[10]) > 20000]))
    buildAllStudentsProfiles()

if __name__ == "__main__":
    main()
