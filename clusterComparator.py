from scipy import spatial
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import main


def getNeighbors(targetStudentID,u,h,r,a, all_students_profiles):

    onlyIDs = []  # saves all the students' Ids
    for l in [u,h,r,a]:
        onlyIDs.append([tup[0] for tup in l])

    index = [onlyIDs.index(l) for l in onlyIDs for el in l if targetStudentID == el]  # discover the profile of the student
    #print("O target student", targetStudentID ," encontra-se no cluster: ", index)
    if index[0] != 3:  # if the target student does not belong to the achievers cluster

        # students who are in the profile above of the target Student's profile
        #students = [el for el in onlyIDs[index[0]+1] if veryfyStudentYear(el, all_students_profiles, "testSet")]

        students = [el for el in onlyIDs[index[0]+1]]

        students_profiles = [all_students_profiles.get(studentId) for studentId in students]
        #print("neighbors do cluster acima:", students)

    else:
        # return all neighbors from the same cluster except himself
        #students = [el for el in onlyIDs[index[0]] if veryfyStudentYear(el, all_students_profiles,  "testSet")] # students who are in the profile same profile as the target student and did the course before 2018
        students = [el for el in onlyIDs[index[0]]]

        students_profiles = [all_students_profiles.get(studentId) for studentId in students if studentId != targetStudentID]
        #print("student neighbor profiles : ", students_profiles )

    return students_profiles, index[0]


'''
# scrutinizes the data and makes some calculations to find the k nearest neighbors
def scrutinizeData(neighbors_profiles, s, ba, q, bo, dic_cosine={}, dic_pearson={}, dic_spearman={}):

    if len(neighbors_profiles) > 0:
        neighbor_profile=neighbors_profiles[0]
        neighbor_items_description = neighbor_profile.getStudentItemsDescription() # gets data from the evaluation item table
        neighbor_skills = ([el for el in neighbor_items_description if el[4] == "Skill"]) # sort by the first element of list, which is the date
        badges = sorted([el for el in neighbor_items_description if el[4] == "Badge"])
        quizzes = sorted([el for el in neighbor_items_description if el[4] == "Quiz"])
        bonus = sorted([el for el in neighbor_items_description if el[4] == "Bonus"])

        evaluationItems = neighbor_profile.getStudentEvaluationItems()

        if len(neighbor_skills) > 1 and len(s) > 1: # just if the target student and the neighbor have earned more than one skill

             # estas medidas têm em conta somente o itemID e calculam distancias entre os itemsIDs.
            #calculate cosine similarity
            cosine = 1 - spatial.distance.cosine([el[0] for el in new_l], [el[1] for el in new_l])
            #key = cosine distance | value = student's skills
            dic_cosine[neighbor_profile] = [neighbor_skills, cosine]

            #calculate pearson correlation
            pearson = stats.pearsonr([el[0] for el in new_l], [el[1] for el in new_l])
            dic_pearson[neighbor_profile] = [neighbor_skills, pearson[0]]

            #calculate spearman correlation
            spearman = stats.spearmanr([el[0] for el in new_l], [el[1] for el in new_l])
            dic_spearman[neighbor_profile] = [neighbor_skills, spearman[0]]


        return scrutinizeData(neighbors_profiles[1:], s, ba, q, bo, dic_cosine, dic_pearson, dic_spearman)

    else:
        return dic_cosine, dic_pearson, dic_spearman
        
'''

def veryfyStudentYear(studentID, all_students_profiles, tag):
    student_profile = all_students_profiles.get(studentID)
    l = student_profile.getStudentResults()

    if tag == "trainSet":
        for el in l:  # normalmente a lista l só deve ter um elemento a não ser que haja um aluno que esteve inscrito na cadeira dois anos diferentes
            if float(el[0]) < 2018:
                return True
    else:
        for el in l:  # normalmente a lista l só deve ter um elemento a não ser que haja um aluno que esteve inscrito na cadeira dois anos diferentes
            if float(el[0]) == 2017:
                return True

    return False

