#####################
# Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz.

# before: 'hi my name is john i am learning python'
# after: 'hİ mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN'


#Enumerate: Otomatik Counter/Indexer ile for
sentence = 'hi my name is john i am learning python'
def upper_odd_indexes(any):
    new_sentence = ''
    for i, char in enumerate(any):
        if i % 2 == 0:
            new_sentence += char.upper()
        else:
            new_sentence += char.lower()
    return new_sentence
        
            

    
# print(upper_odd_indexes(sentence))


###################################

#divide_students fonksiyonu yazınız
#çift indexte yer alan öğrencileri bir listeye alınız.
#tek indexte yer alan öğrencileri başka bir listeye alınız
#fakat bu iki liste tek bir liste olarak return olsun.

# students = ['John','Mark','Vanessa','Mariam']

# def divide_students():
#     groups = [[],[]]
#     for i, student in enumerate(students):
#         if i % 2 == 0:
#             groups[0].append(student)
#         else:
#             groups[1].append(student)
#     print(groups)
    
# st = divide_students()



#######################
# Zip 
#######################
students = ['John','Mark','Vanessa','Mariam']
departments = ['mathematics','statistics','physics','astronomy']
ages = [23,30,26,22]
a= list(zip(students,departments,ages))

print(a)

######################
# lambda, map, filter, reduce
######################
#lambda
# def summer(a,b):
#     return a+b

# # summer(1,3)*9
# # print(summer(1,3)*9)
# new_sum = lambda a,b: a+b
# new_sum(3,4)

# map

salaries = [1000,2000,3000,4000,5000]

def new_salary(x):
    return x*20/ 100+ x

# new_salary(1000)

# for salary in salaries:
#     print(new_salary(salary))

# Bunun yerine map fonksiyonu kullanabiliriz
a=list(map(new_salary,salaries))
print(a)

#lambda ile map ilişkisi
# b=list(map(lambda x: x*20/100+x,salaries))
# print(b)
# b = [1200.0, 2400.0, 3600.0, 4800.0, 6000.0]
#map derki bana bir fonksiyon ve bir de liste ver.
#o listede ilk indexten başlayıp tek tek hepsini dolaşarak uygulayabilirim.
# Filter

# list_store = [1,2,3,4,5,6,7,8,9,10]
# c = list(filter(lambda x: x%2 == 0,list_store))
# print(c)
#c = [2, 4, 6, 8, 10]

# Reduce
from functools import reduce
list_store = [1,2,3,4]
d = reduce(lambda a,b:a+b,list_store)
print(f'd:{d}')


#########################################
#COMPREHENSIONS # ÖNEMLİ BİR KONU #
#########################################

# List Comprehensions

# Klasik Yontem
# null_list = []
def new_salary(x):
    return x*20/ 100+ x
salaries = [1000,2000,3000,4000,5000]

# for salary in salaries:
#     if salary > 3000:
#         null_list.append(new_salary(salary))
#     else:
#         null_list.append(new_salary(salary*2))


# print(null_list)  

# List Comprehension yardımıyla yukarıdaki işlemi aşağıdaki gibi yapabiliriz

# new_list=[new_salary(salary*2) if salary <3000 else new_salary(salary) for salary in salaries]
# print(new_list)

# students = ['John','Mark','Vanessa','Mariam']

# students_no = ['John','Vanessa']

# a=[student.lower() if student in students_no else student.upper() for student in students ]
# print(a)

# Bu yapı sayesinde hem bir listeyi bir fonksiyon yardımıyla for döngüsünde içinde gezip aynı zamanda if else ile sorgulayabiliriz.
# Sadece if kullanacaksak listenin en sağında olmak zorunda
#if else birlikte kullanılacak ise for döngüsünün solunda kullanabiliriz.

# Dictionary Comprehensions

# dictionary = { 'a' : 1,
#                'b' : 2,
#                'c' : 3,
#                'd' : 4}

# dictionary.keys() #dictionary key'lerine ulaşır
# dictionary.values() #dictionary value'lerine ulaşır
# dictionary.items() #hem key hem value ulaşır

# print({k:v**2 for (k,v) in dictionary.items()})
# print({k.upper():v**2  for (k,v) in dictionary.items() if v %2 == 0})
# print({k.upper():v**2 if v %2 == 0 else  k.lower() and v**2 for (k,v) in dictionary.items() })


###############################
# Uygulama - Mülakat Sorusu
###############################

# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir
# numbers = range(0,10)
# new_dict = {}

# print({i:i**2 for i in numbers if i %2 == 0})



#################################
# List & Dict Comprehension Uygulamalar
#################################

#################################
# Bir Veri Setindeki Değişken İsimlerini Değiştirmek
#################################



# before:
# ['total','speeding','alcohol','not_distracted','no_previous','ins_premium','ins_losses','abbrev']

# after
# ['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES', 'ABBREV']


# a=['total','speeding','alcohol','not_distracted','no_previous','ins_premium','ins_losses','abbrev']
# b=[i.upper() for i in a ]
# print(b)

import seaborn as sns
# df = sns.load_dataset('car_crashes')

# print(df.columns)
# A=[]
# for col in df.columns:
#     A.append(col.upper())
# df.columns = A
# print(df.columns)

#List compherision çözümü
# df = sns.load_dataset('car_crashes')
# df.columns = [col.upper() for col in df.columns]
# # print(df.columns)

# ###############################################
# # İsminde 'INS' olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.
# df.columns = ['FLAG_'+ col if 'INS' in col else 'NO_FLAG_'+ col for col in df.columns]
# print(df.columns)

#######################
# Amaç key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak.
# Sadece sayısal değişkenler için yapmak istiyoruz.
#######################

# Output:
# {'total': ['mean', 'min', 'max', 'var'],
#  'speeding': ['mean', 'min', 'max', 'var'],
#  'alcohol': ['mean', 'min', 'max', 'var'],
#  'not_distracted': ['mean', 'min', 'max', 'var'],
#  'no_previous': ['mean', 'min', 'max', 'var'],
#  'ins_premium': ['mean', 'min', 'max', 'var'],
#  'ins_losses': ['mean', 'min', 'max', 'var']}

df = sns.load_dataset('car_crashes')
df.columns

num_cols = [col for col in df.columns if df[col].dtype != 'O']
agg_list =['mean', 'min', 'max', 'var']
new_dict={col:agg_list for col in num_cols}
# print(x)
print(df[num_cols].agg(new_dict))
'''
          total  speeding    alcohol  not_distracted  no_previous   ins_premium  ins_losses
mean  15.790196  4.998196   4.886784       13.573176    14.004882    886.957647  134.493137
min    5.900000  1.792000   1.593000        1.760000     5.900000    641.960000   82.750000
max   23.900000  9.450000  10.038000       23.661000    21.280000   1301.520000  194.780000
var   16.990902  4.071303   2.989901       20.330874    14.172755  31789.565170  616.823046

yukarıdaki metod (ag) ile dataframe(df) içerisindeki numerik kolonlara (num_cols) yeni sözlük yapımızdaki işlemleri uygulayabiliriz.
'''