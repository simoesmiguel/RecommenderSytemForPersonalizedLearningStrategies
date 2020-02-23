from functools import reduce
import collections
import main
from Levenshtein import _levenshtein


def checkDateinrange(neighbor_skills, evaluationItems):
    all_skills_in_range = {}
    in_range = False
    already_found = False
    for el1 in neighbor_skills:
        item_id = el1[0]
        for el2 in evaluationItems:
            if item_id == el2[2]:  # se os item_id coincidirem
                already_found = True
                date_id = el2[0]
                formatted_date_id = date_id[2:4] + "/" + date_id[4:6] + "/" + date_id[6:8]
                if main.checkDates(formatted_date_id):  # checkar se a data está na primeira metade do semestre

                    all_skills_in_range[date_id] = el1
                    break

    return collections.OrderedDict(
        sorted(all_skills_in_range.items()))  # retorna por ordem da data (do mais antigo para o mais recente)



# scrutinizes the data and makes some calculations to find the k nearest neighbors
def scrutinizeData(neighbors_profiles, s, ba, q, bo, dic_editdistance={}, dic_euclideandistance={}):
    if len(neighbors_profiles) > 0:
        neighbor_profile = neighbors_profiles[0]
        neighbor_items_description = neighbor_profile.getStudentItemsDescription()  # gets data from the evaluation item table
        neighbor_skills = ([el for el in neighbor_items_description if
                            el[4] == "Skill"])  # sort by the first element of list, which is the date
        badges = sorted([el for el in neighbor_items_description if el[4] == "Badge"])
        quizzes = sorted([el for el in neighbor_items_description if el[4] == "Quiz"])
        bonus = sorted([el for el in neighbor_items_description if el[4] == "Bonus"])

        evaluationItems = neighbor_profile.getStudentEvaluationItems()

        if len(neighbor_skills) > 1 and len(
                s) > 1:  # just if the target student and the neighbor have earned more than one skill

            all_skills_in_range = checkDateinrange(neighbor_skills, evaluationItems)
            lista = [all_skills_in_range[key] for key in all_skills_in_range]

            if lista != []:
                # print("skills do neighbor: ",len(all_skills_in_range))
                if len(lista) >= len(
                        s):  # we don't consider the neighbors who have less completed skills than the target student
                    just_codes = [el[11] for el in lista]


                    # =============================================
                    '''this piece of code joins the skills which were achieved both by target student and its neighbor, but were not 
                    achieved in the same order. For instance:
                    
                    student_target_skills = [1,4,2,5,6]
                    neighbor_skills =       [3,4,1,7,8]
                    
                    by the end of this code, new_l = [(1,1),(4,4),(2,3),(5,7),(6,8)]
                    '''
                    copy_s = [el for el in s]
                    copy_just_codes = [el for el in just_codes]
                    n = []  # tuples list
                    elements_to_delete1 = []
                    elements_to_delete2 = []

                    for code in copy_s:
                        for code2 in just_codes:
                            if code == code2:
                                elements_to_delete1.append(code)
                                elements_to_delete2.append(code2)
                                n.append((code, code2))
                                break

                    for code1 in elements_to_delete1:
                        copy_s.remove(code1)
                    for code2 in elements_to_delete2:
                        just_codes.remove(code2)

                    n2 = list(zip(just_codes, copy_s))
                    new_l = n + n2
                    # =============================================

                    # calculate Levenshtein (edit) distance
                    lst_all_distances = [calculateLevenshteinDistance(tupl[0], tupl[1]) for tupl in new_l]

                    avg_distance = reduce(lambda a, b: a + b, lst_all_distances) / len(lst_all_distances)
                    dic_editdistance[neighbor_profile] = [avg_distance, copy_just_codes]

        return scrutinizeData(neighbors_profiles[1:], s, ba, q, bo, dic_editdistance, dic_euclideandistance)

    else:
        return dic_editdistance, dic_euclideandistance

# Levenshtein Distance
def calculateLevenshteinDistance(string1, string2):
    return _levenshtein.distance(string1, string2)
