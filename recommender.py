
'''
returns n_skills_to_recommend taking into account the number of knwighbors
'''

def recommendSkills(list_skills, target_student_skills, kneighbors, n_skills_to_recommend):

    # go through all the k nearest neighbors and save the skills that were not yet performed by the target student
    skills_to_recommend = []
    neighbors=1

    for tupl in list_skills:
        if neighbors <= kneighbors or len(skills_to_recommend) == 0: # if the number of neighbors visited is less than
                                                                    # the number of neighbors desired to search, or if any recommendation was not found yet
            all_codes = tupl[1]
            for code in all_codes:
                if code not in target_student_skills:
                    skills_to_recommend.append(code)
            neighbors+=1
        else:
            break


    code_ocurrences =[] # this list saves the number of ocurrences of each code
    for code in skills_to_recommend:
        if code not in [el[0] for el in code_ocurrences]:
            code_ocurrences.append((code, skills_to_recommend.count(code)))

    #print(code_ocurrences)

    final_list = [ tupl for tupl in reversed(sorted(code_ocurrences, key=lambda item: item[1]))] # sort the list by the number of ocurrences
    #print("list to recomend", final_list)

    if len(final_list) <= n_skills_to_recommend:
        return [el[0] for el in final_list]
    else:
        return [el[0] for el in final_list][0:n_skills_to_recommend]



def recommendSkills2(list_skills, target_student_info, kneighbors, n_recommendations):

    # go through all the k nearest neighbors and save the skills that were not yet performed by the target student
    skills_to_recommend, badges_to_recommend, quizzes_to_recommend, bonus_to_recommend, posts_to_recommend = [], [], [], [], []
    neighbors=1

    for tupl in list_skills:
        if neighbors <= kneighbors or (len(skills_to_recommend) == 0 and len(badges_to_recommend)== 0 and len(quizzes_to_recommend)== 0 and len(bonus_to_recommend)== 0 and len(posts_to_recommend)== 0):

            # if the number of neighbors visited is less than the number of neighbors desired to search, or if any recommendation was not found yet
            all_info = tupl[1]

            for tuple in all_info:
                if tuple[0] == "skills":
                    for code in tuple[1]:
                        if code not in target_student_info[0]:
                            skills_to_recommend.append(code)

                elif tuple[0] == "badges":
                    for code in tuple[1]:
                        if code not in target_student_info[1]:
                            badges_to_recommend.append(code)

                elif tuple[0] == "bonus":
                    for code in tuple[1]:
                        if code not in target_student_info[2]:
                            bonus_to_recommend.append(code)

                elif tuple[0] == "quizzes":
                    for code in tuple[1]:
                        if code not in target_student_info[3]:
                            quizzes_to_recommend.append(code)
                else:
                    for code in tuple[1]:
                        if code not in target_student_info[4]:
                            posts_to_recommend.append(code)

            neighbors+=1
        else:
            break


    a = list_occurrences(skills_to_recommend, n_recommendations)
    b = list_occurrences(badges_to_recommend, n_recommendations)
    c = list_occurrences(bonus_to_recommend, n_recommendations)
    d = list_occurrences(quizzes_to_recommend, n_recommendations)
    e = list_occurrences(posts_to_recommend, n_recommendations)

    return [a, b, c, d, e]


def list_occurrences(recommendations, n_skills_to_recommend):
    code_ocurrences = []  # this list saves the number of ocurrences of each element of the list "recommendations"
    for code in recommendations:
        if code not in [el[0] for el in code_ocurrences]:
            code_ocurrences.append((code, recommendations.count(code)))

    final_list = [ tupl for tupl in reversed(sorted(code_ocurrences, key=lambda item: item[1]))] # sort the list by the number of ocurrences

    if len(final_list) <= n_skills_to_recommend:
        return [el[0] for el in final_list]
    else:
        return [el[0] for el in final_list][0:n_skills_to_recommend]