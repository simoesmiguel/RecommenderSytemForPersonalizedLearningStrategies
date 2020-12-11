
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

from statistics import median


f = open("./outputFile_10_11", "r")

to_print =[]


all_ids = []

dic ={}

count =100

xps =""

for line in f:
    if count !=0 and count < 3:
        to_print.append(line)
        if count==1:
            xps = line.split(":::")[1].strip()
        elif count==2:
            n_activities = line.split(":::")[1].strip()
            if n_activities in dic:
                dic[n_activities].append(xps)
            else:
                dic[n_activities] = [xps]

        count +=1

    elif count == 3:
        to_print.append("\n")
        xps =""
        count =0

    if "Student ID:" in line:
        to_print.append(line)
        all_ids.append(line.split(":")[1].strip())
        count =1


# for line in to_print:
#	print(line)




def drawBoxPlot(d):

    all_groups = []

    for k in d:
        g = pd.DataFrame({'group': np.repeat(str(k), len(d.get(str(k)))), 'value': [float(n) for n in d.get(str(k))]})

        all_groups.append(g)

    for el in all_groups:
        print(el)

    df = all_groups[0].append(all_groups[1]).append(all_groups[2]).append(all_groups[3]).append(all_groups[4]).append(all_groups[5]).append(all_groups[6]).append(all_groups[7]).append(all_groups[8])

    # Usual boxplot
    sns.boxplot(x='group', y='value', data=df)



    #ax = sns.boxplot(x='group', y='value', data=df)
    # ax = sns.stripplot(x='group', y='value', data=df, color="orange", jitter=0.2, size=2.5)

    # Calculate number of obs per group & median to position labels
    # medians = df.groupby(['group'])['value'].median().values
    #all = [u, h, r, ac]

    #medians = [median(l) for l in dic.keys() if l!= []]
    #nobs = ["n: " + str(len(l)) for l in [u, h, r, ac] if l != []]  # number of observations

    # Add it to the plot
    #pos = range(len(nobs))
    #for tick, label in zip(pos, ax.get_xticklabels()):
        #plt.text(pos[tick], medians[tick] + 0.4, nobs[tick], horizontalalignment='center', size='medium', color='black',
         #        weight='semibold')

    plt.ylabel("Difference in XPs")
    plt.xlabel("Number of activities carried out")

    plt.title("Difference in XPs as a function of the number of activities carried out by the students")

    plt.show()

def orderdicbyKey(dic):
    return {k: v for k, v in sorted(dic.items(), key=lambda item: int(item[0]))}

#drawBoxPlot(dic)

dic2 = {"1":["150","200"], "2":["150", "220"] , "3":["150","220","250","500"], "4":["220","250","520"],"5":["250","400", "450", "530"],"6":[ "450", "500", "600", "800", "900"], "7":["750", "850", "1200", "1600"], "8":["750","1200", "1500"], "9": ["800", "1200","1450"]}

print(orderdicbyKey(dic2))

drawBoxPlot(orderdicbyKey(dic2))
