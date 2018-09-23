# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the list of Voters
dataset1=pd.read_csv('45.csv')
dataset1['EName'] = dataset1['EName'].str.lower()
dataset1['EName'] = dataset1['EName'].str.strip()
dataset1['RName'] = dataset1['RName'].str.lower()
dataset1['RName'] = dataset1['RName'].str.strip()
dataset1['RName'] = dataset1['RName'].fillna('xyz') # filling missing values



#import re 
sirname=[]
for i in range (0,181621):
    """a = re.sub(r".*?(\w+)\s*$", r"\1", dataset1['Name'][i])"""
    A=dataset1['EName'][i].split()
    X=dataset1['RName'][i].split()
    a=A[-1]
    b=A[0]
    if a==b:
        a_s=X[-1]# relative sirname
        b_s=X[0]
        if a_s==b_s:
            
            continue
        else:
            sirname.append(a_s) 
    else:
        sirname.append(a)
        




    
    
# Importing the dataset
dataset = pd.read_csv('surnames_all.csv')
dataset['Sirname'] = dataset['Sirname'].str.lower()
dataset['Sirname'] = dataset['Sirname'].str.strip()


#for unique values 
unique=np.array(sirname)
unique= np.unique(unique)


#sirname list range
gujars=20      
Punjabi=262
Brahmin=291
Muslim=643
Sikh=1084
jaat=1389
SC=1455


list2=[]
list3=[]
p=len(sirname)
for a in range (0,100):
    x=sirname[a]
    for j in range (0,1455):
        b=dataset['Sirname'][j]
        if  x==b:
            if j in range (0,gujars) :
                list2.append(1)
            elif j in range (gujars,Punjabi) :
                list2.append(2)
            elif j in range (Punjabi,Brahmin) :
                list2.append(3)
            elif j in range (Brahmin,Muslim) :
                list2.append(4)
            elif j in range (Muslim,Sikh) :
                list2.append(5)
            elif j in range (Sikh,jaat) :
                list2.append(6)
            elif  j in range (jaat,SC) :
                list2.append(7)
        else:
            continue



B=list2.count(1) 
C=list2.count(2) 
D=list2.count(3) 
E=list2.count(4) 
F=list2.count(5) 
G=list2.count(6)
H=list2.count(7)

A=p-len(list2)

other_index=list2.index(A)
          
                
# Data to plot
labels = 'Others', 'gujars', 'Punjabi(Hindu)', 'Brahmin','Muslim','Sikh','jaat','SC-St'
sizes = [A,B,C,D,E,F,G,H]
colors = ['yellow', 'yellowgreen', 'lightcoral', 'lightskyblue','magenta','green','blue',"cyan"]
explode = (0, 0, 0, 0,0,0,0,0.1)  # explode last slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
#plt.legend(labels,loc="best")
plt.show()  


"""# Importing the dataset
dataset3 = pd.read_csv('unique_sirname.csv')
i=0
list1=[]
list2=[]
for i in range(0,10108):
    a=dataset3['sirname'][i]
    list1.append(a)
for j in range (0,1455):
    b=dataset['Sirname'][j]
    list2.append(b)

unique_list=np.array(list(set(list1) - set(list2)))"""



        



              
            
            
            

