from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

def getNeighbors(targetStudentID,u,h,r,a, all_students_profiles):

    onlyIDs = []  # saves all the students' Ids
    for l in [u,h,r,a]:
        onlyIDs.append([tup[0] for tup in l])

    index =[onlyIDs.index(l) for l in onlyIDs for el in l if targetStudentID == el]  # discover the profile of the student
    print("O target student encontra-se no cluster: ", index)
    if index[0] != 3:  # if the target student does not belong to the achievers cluster
        students = onlyIDs[index[0]+1]  # students who are in the profile above of the target Student's profile

        students_profiles = [all_students_profiles.get(studentId) for studentId in students]
        print("neighbors do cluster acima:", students)

        '''
        first_s = students_profiles[0]
        print(first_s.getStudentResults(),"\n")
        print(first_s.getStudentEvaluationItems(),"\n")
        print(first_s.getStudentItemsDescription(),"\n")
        print(len(first_s.getStudentEvaluationItems()), "\n")
        print(len(first_s.getStudentItemsDescription()), "\n")

        print(first_s.getMoodleParticipation(),"\n")
        print(first_s.getPosts(),"\n")
        print(first_s.getMessageAnalysis(),"\n")
        '''
        return students_profiles

# scrutinizes the data and makes some calculations to find the k nearest neighbors
def scrutinizeData(neighbors_profiles, s, ba, q, bo, dic={}):

    if len(neighbors_profiles) > 0:
        neighbor_profile=neighbors_profiles[0]
        neighbor_items_description = neighbor_profile.getStudentItemsDescription()
        neighbor_skills = sorted([int(el[0]) for el in neighbor_items_description if el[2] == "Skill"]) # sort by the first element of list, which is the date
        badges = sorted([el for el in neighbor_items_description if el[2] == "Badge"])
        quizzes = sorted([el for el in neighbor_items_description if el[2] == "Quiz"])
        bonus = sorted([el for el in neighbor_items_description if el[2] == "Bonus"])

        if len(neighbor_skills) > 1 and len(s) > 1: # só se o aluno tiver mais do que uma skill já feita é que se vai comparar com os vizinhos
            new_l = list(zip(neighbor_skills, s))

            cosine = 1 - spatial.distance.cosine([el[0] for el in new_l], [el[1] for el in new_l])

            #key = cosine distance | value = student's skills
            dic[cosine] = [neighbor_skills, neighbor_profile]

        return scrutinizeData(neighbors_profiles[1:], s, ba, q, bo, dic)

    else:
        return dic








