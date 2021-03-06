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
        if lista != []: # se o neighbor tiver completado skills até 15/04
            just_codes = [el[11] for el in lista]
            new_l = [(code, code) for code in just_codes]
            return new_l, just_codes

    return [],[]


def auxiliar2(neighbor_posts, target_student_posts, content_topic):

    '''
        target_student_posts -> target student's posts that were made until 15/04
            type: dictionary
        neighbor_posts -> posts that the student made without time range, i.e all posts that the neighbor student did
            type: list of lists where each list is a line represents one post from the neighbor student
    '''


    posts_in_range = []
    topic_dic_ordered = []
    topic_dic = {}
    for lista in neighbor_posts:
        date = lista[2][2:4] + "/" + lista[2][4:6] + "/" + lista[2][6:8]
        if main.checkDates(date):
            posts_in_range.append(lista)

    if posts_in_range != []:  # if the neighbor student has made any posts up to 15/04

        discussion_topics = [l[0] for l in content_topic for lista in posts_in_range if lista[3] == l[-1]]
        for el in discussion_topics:
            if el not in topic_dic.keys():
                topic_dic[el] = 1
            else:
                topic_dic[el] = topic_dic.get(el) + 1

        #topic_dic_ordered = sorted(topic_dic.items())

        a =[key+str(target_student_posts.get(key)) for key in target_student_posts]  # ex: ["BugsForum2", "Questions1", "..."]
        b = [key+str(topic_dic.get(key)) for key in topic_dic]   # ex: ["BugsForum2", "Questions1", "..."]

        if len(a) == 0: # se o target student ainda não tiver nenhum post
            new_l = [(post, post) for post in b]

        elif len(a) > 0 and len(a) < len(b): # se o target student já tiver mais do que um post e o neighbor tiver mais posts do que o target student
            new_l = orderElements(a, b)

        else:
            new_l=[]

        return new_l, b

    else:
        return [],[]




# scrutinizes the data and makes some calculations to find the k nearest neighbors
def scrutinizeData(neighbors_profiles, s, ba, q, bo, topic_dic, content_topic):

    global neighbor_indicators
    global target_student_indicators


    target_student_indicators = [1,1,1,1,1] # skills, badges, bonus, quizzes, posts

    if len(s) == 0:
        target_student_indicators[0] = 0
    if len(ba) == 0:
        target_student_indicators[1] = 0
    if len(bo) == 0:
        target_student_indicators[2] = 0
    if len(q) == 0:
        target_student_indicators[3] = 0
    if len(topic_dic.items()) == 0:
        target_student_indicators[4] = 0


    dic_skills, dic_badges, dic_quizzes, dic_bonus, posts_list = [], [], [], [], []
    lista_all = []


    for profile in neighbors_profiles: # for all the target student's neighbors

        neighbor_indicators = [0, 0, 0, 0, 0]

        neighbor_items_description = profile.getStudentItemsDescription()  # gets data from the evaluation item table
        neighbor_skills = ([el for el in neighbor_items_description if
                            el[4] == "Skill"])  # sort by the first element of list, which is the date
        neighbor_badges = [el for el in neighbor_items_description if el[4] == "Badge"]
        neighbor_quizzes = [el for el in neighbor_items_description if el[4] == "Quiz"]
        neighbor_bonus = [el for el in neighbor_items_description if el[4] == "Bonus"]

        evaluationItems = profile.getStudentEvaluationItems()

        neighbor_posts = profile.getPosts()


        if len(neighbor_skills) != 0:
            neighbor_indicators[0] = 1
        if len(neighbor_badges) != 0:
            neighbor_indicators[1] = 1
        if len(neighbor_bonus) != 0:
            neighbor_indicators[2] = 1
        if len(neighbor_quizzes) != 0:
            neighbor_indicators[3] = 1
        if len(neighbor_posts) != 0:
            neighbor_indicators[4] = 1


        avg_distance_skills, avg_distance_badges, avg_distance_bonus, avg_distance_quizzes, avg_distance_posts = -1, -1, -1, -1, -1

        ##neighbor skills
        new_l, codes_skills = auxiliar(neighbor_skills, s, evaluationItems)
        if new_l != [] and codes_skills != []:
            avg_distance_skills = calculateAvgDistance(new_l)

        ##neighbor badges
        new_l, codes_badges = auxiliar(neighbor_badges, ba, evaluationItems)
        if new_l != [] and codes_badges != []:
            avg_distance_badges = calculateAvgDistance(new_l)


        ##neighbor bonus
        new_l, codes_bonus = auxiliar(neighbor_bonus, bo, evaluationItems)
        if new_l != [] and codes_bonus != []:
            avg_distance_bonus = calculateAvgDistance(new_l)

            ##neighbor quizzes
        new_l, codes_quizzes = auxiliar(neighbor_quizzes, q, evaluationItems)
        if new_l != [] and codes_quizzes != []:
            avg_distance_quizzes = calculateAvgDistance(new_l)


        new_l2, n_posts_encoded = auxiliar2(neighbor_posts, topic_dic, content_topic )
        ##neighbor posts
        if new_l2 != [] and  n_posts_encoded !=[]:
            avg_distance_posts = calculateAvgDistance(new_l2)


        count1, count2 = 0, 0
        for i in range(len(target_student_indicators)):
            if target_student_indicators[i] == 1:
                count1 += 1
                if neighbor_indicators[i] == 1:
                    count2 += 1

        if count2 == count1: # we found a neighbor of the target student
            all_distances = [avg_distance_skills, avg_distance_badges, avg_distance_bonus, avg_distance_quizzes, avg_distance_posts]
            total_distance=0
            for el in all_distances:
                if el != -1:
                    total_distance+=el

            lista_all.append( ( total_distance, [("skills", codes_skills ),("badges", codes_badges),("bonus", codes_bonus),("quizzes", codes_quizzes),("posts",n_posts_encoded)]))


    return lista_all





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