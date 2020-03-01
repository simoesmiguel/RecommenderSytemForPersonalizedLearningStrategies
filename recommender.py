
'''
returns n_skills_to_recommend taking into account the number of knwighbors
'''

def recommendSkills(dic_skills, target_student_skills, kneighbors, n_skills_to_recommend):

    # go through all the k nearest neighbors and save the skills that were not yet performed by the target student
    skills_to_recommend = []
    neighbors=1

    for tupl in dic_skills:
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