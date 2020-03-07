from functools import reduce
import collections
import main
from Levenshtein import _levenshtein


#dic_skills, dic_badges, dic_quizzes, dic_bonus=[], [], [], []

def checkDateinrange(neighbor_skills, evaluationItems):
    all_skills_in_range = {}
    for el1 in neighbor_skills:
        item_id = el1[0]
        for el2 in evaluationItems:
            if item_id == el2[2]:  # se os item_id coincidirem
                date_id = el2[0]
                formatted_date_id = date_id[2:4] + "/" + date_id[4:6] + "/" + date_id[6:8]
                if main.checkDates(formatted_date_id):  # checkar se a data está na primeira metade do semestre

                    all_skills_in_range[date_id] = el1
                    break

    return collections.OrderedDict(
        sorted(all_skills_in_range.items()))  # retorna por ordem da data (do mais antigo para o mais recente)


def calculateAvgDistance(new_l):

    # calculate Levenshtein (edit) distance
    lst_all_distances = [calculateLevenshteinDistance(tupl[0], tupl[1]) for tupl in new_l]
    # calculate avg distance
    avg_distance = reduce(lambda a, b: a + b, lst_all_distances) / len(lst_all_distances)

    return avg_distance

def auxiliar(neighbor_skills, target_student_skills, evaluationItems):

    if len(neighbor_skills) > 1 and len(
            target_student_skills) >= 1:  # just if the target student and the neighbor have earned more than one skill

        all_skills_in_range = checkDateinrange(neighbor_skills, evaluationItems)
        lista = [all_skills_in_range[key] for key in all_skills_in_range]

        if lista != []:
            # print("skills do neighbor: ",len(all_skills_in_range))
            if len(lista) >= len(
                    target_student_skills):  # we don't consider the neighbors who have less completed skills than the target student
                just_codes = [el[11] for el in lista]

                new_l = orderElements(target_student_skills, just_codes)
                return new_l, just_codes

    elif len(target_student_skills) == 0: # se o aluno ainda não tiver nenhumas skills
        all_skills_in_range = checkDateinrange(neighbor_skills, evaluationItems)
        lista = [all_skills_in_range[key] for key in all_skills_in_range]
        if lista != []:
            just_codes = [el[11] for el in lista]

            new_l = [(code, code) for code in just_codes]
            return new_l, just_codes

    return [],[]




# scrutinizes the data and makes some calculations to find the k nearest neighbors
def scrutinizeData(neighbors_profiles, s, ba, q, bo):
    #global dic_skills, dic_badges, dic_quizzes, dic_bonus
    dic_skills, dic_badges, dic_quizzes, dic_bonus = [], [], [], []

    for profile in neighbors_profiles:

        neighbor_items_description = profile.getStudentItemsDescription()  # gets data from the evaluation item table
        neighbor_skills = ([el for el in neighbor_items_description if
                            el[4] == "Skill"])  # sort by the first element of list, which is the date
        neighbor_badges = [el for el in neighbor_items_description if el[4] == "Badge"]
        neighbor_quizzes = [el for el in neighbor_items_description if el[4] == "Quiz"]
        neighbor_bonus = [el for el in neighbor_items_description if el[4] == "Bonus"]

        evaluationItems = profile.getStudentEvaluationItems()

        new_l, codes = auxiliar(neighbor_skills, s, evaluationItems)
        if new_l != [] and codes != []:
            #dic_skills = saveDictionary(dic_skills, dic_badges, dic_quizzes, dic_bonus, new_l, neighbor_profile, codes, "skills")
            avg_distance = calculateAvgDistance(new_l)
            if (avg_distance, codes) not in dic_skills:
                dic_skills.append((avg_distance, codes))

        new_l, codes = auxiliar(neighbor_badges, ba, evaluationItems)
        if new_l != [] and codes != []:
            avg_distance = calculateAvgDistance(new_l)
            if (avg_distance, codes) not in dic_badges:
                dic_badges.append((avg_distance, codes))

        new_l, codes = auxiliar(neighbor_quizzes, q, evaluationItems)
        if new_l != [] and codes != []:
            avg_distance = calculateAvgDistance(new_l)
            if (avg_distance, codes) not in dic_quizzes:
                dic_quizzes.append((avg_distance, codes))

        new_l, codes = auxiliar(neighbor_bonus, bo, evaluationItems)
        if new_l != [] and codes != []:
            avg_distance = calculateAvgDistance(new_l)
            if (avg_distance, codes) not in dic_quizzes:
                dic_quizzes.append((avg_distance, codes))


    return dic_skills, dic_badges, dic_quizzes, dic_bonus


# Levenshtein Distance
def calculateLevenshteinDistance(string1, string2):
    return _levenshtein.distance(string1, string2)

def calculateEuclideanDistance(string1, string2):
    pass


def orderElements(tg_student_act, neighbor_act):
    '''this function joins the skills which were achieved both by target student and its neighbor, but were not
       achieved in the same order. Also, this function joins the most similar skills between the target student and
       its neighbor in order to minimize the avg distance
       For instance:

       student_target_skills = [aa,bb,cd,ef,zx]
       neighbor_skills =       [ce,bb,aa,fg,jr]

       by the end of this code, new_l = [(aa,aa),(bb,bb),(cd,ce),(ef,fg),(zx,jr)]
       Attention! Most of the times, neighbor_skills is a bigger array than student_target_skills
    '''

    copy_act = [el for el in tg_student_act]  # all elements except the students' level
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
                dist = calculateLevenshteinDistance(el, el2)
                dic[(el, el2)] = dist

        ordered_dic = orderdicbyValue(dic)
        most_similar_tuple = ordered_dic.popitem()[0]
        already_added.append(most_similar_tuple[1])

        n2.append(most_similar_tuple)

    new_l = n + n2

    return new_l

def orderdicbyValue(dic): # orders dict in reverse order
    return {k: v for k, v in reversed(sorted(dic.items(), key=lambda item: item[1]))}