###############################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
###############################################
# - NumPy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)

#############################################
# NUMPY
#############################################

# Neden NumPy? (Why Numpy?)
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
# Matematiksel İşlemler (Mathematical Operations)

#############################################
# Neden NumPy? Bu dosya
#############################################

import numpy as np

#Eski python çözümü iki arrayin çarpımı
# a = [1,2,3,4]
# b = [2,3,4,5]

# ab = []
# for i in range(0,len(a)):
#     ab.append([a[i]*b[i]])

# # Numpy Çözümüm 

# a = np.array([1,2,3,4])
# b = np.array([2,3,4,5])
# # print(a*b)
# # Daha fonksiyonel daha hızlı daha kolay

# #############################################
# # NumPy Array'i Oluşturmak (Creating Numpy Arrays)
# #############################################
# import numpy as np

# np.array([1, 2, 3, 4, 5])
# type(np.array([1, 2, 3, 4, 5]))
# np.zeros(10, dtype=int)
# np.random.randint(0, 10, size=10)
# np.random.normal(10, 4, (3, 4)) #ortalaması 10 standart sapması 4 3x4 lük normal dağılımlı rastgele array

# print(np.random.normal(10, 4, (3, 4)))

# #############################################
# # NumPy Array Özellikleri (Attibutes of Numpy Arrays)
# #############################################
# import numpy as np

# # ndim: boyut sayısı
# # shape: boyut bilgisi
# # size: toplam eleman sayısı
# # dtype: array veri tipi

# a = np.random.randint(10,size=5)
# print(f'a:{a}')
# print(f'a.ndim:{a.ndim}')
# print(f'a.shape:{a.shape}')
# print(f'a.size:{a.size}')
# print(f'a.dtype:{a.dtype}')

# #############################################
# # Yeniden Şekillendirme (Reshaping)
# #############################################
# import numpy as np

# ar=np.random.randint(1,10,size=9).reshape(3,3)
# print(ar)
#reshape matrisi istediğimiz şekle çevirmemizi sağlar yukarıdaki fonksiyon ile 3 e 3lük matrise çevirdik.

#############################################
# Index Seçimi (Index Selection)
#############################################
# import numpy as np
# a = np.random.randint(10, size=10)
# a[0] # 0. index
# a[0:5] # 0 dan 5 e kadar, (5) dahil değil
# a[0] = 999
# print(a)


# iki boyutlu bir array olursa
# m = np.random.randint(10, size=(3,5))
# print(m)
# print('\n')
# m[2,3] = 1 #[a,b] a = satır indexi b = sütun indexi
# m[:, 0] = 10 # bütün satırlar 0. sütunlar
# m[1, :] = 5 # 1. satır, bütün sütunlar
# m[0:2, 0:3] = 12 # 0 dan 2 ye kadarki satırlar (2) dahil değil, 0 dan 3. sütuna kadarki sütunlar (3) dahil değil
# print(m)

#############################################
# Fancy Index
#############################################
# import numpy as np

# v = np.arange(0, 30, 3)
# # arange ifadesi ardışık artan bir matris için kullanılır. 3 er 3 er artan demek için de parantez içinde en sola 3 yazılır.
# v[1]
# v[4]
# print(v)
# print(v[4])
# catch = [1, 2, 3]
# # fancy index yapısı sayesinde istediğimiz indextekiler tuutabiliriz
# print(v[catch])

#############################################
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
#############################################
# import numpy as np
# v = np.array([1, 2, 3, 4, 5])

# #######################
# # Klasik döngü ile
# #######################
# ab = []
# for i in v:
#     if i < 3:
#         ab.append(i)

# #######################
# # Numpy ile
# #######################
# v < 3

# print(v[v < 3])
# print(v[v > 3])
# print(v[v != 3])
# print(v[v == 3])
# print(v[v >= 3])

# #############################################
# # Matematiksel İşlemler (Mathematical Operations)
# #############################################
# import numpy as np
# v = np.array([1, 2, 3, 4, 5])

# v / 5
# v * 5 / 10
# v ** 2
# v - 1

# np.subtract(v, 1) # arrayden 1 sayısını çıkarmak için
# np.add(v, 1) # arraya 1 eklemek için
# np.mean(v) # arrayin ortalaması
# np.sum(v) # arrayin tüm değerlerinin toplamı
# np.min(v) # arrayin min değeri
# np.max(v) # arrayin max değeri
# np.var(v) # arrayin varyansı
# v = np.subtract(v, 1) 

#######################
# NumPy ile İki Bilinmeyenli Denklem Çözümü
#######################

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

# a = np.array([[5, 1], [1, 3]]) # [[Birinci elemanın katsayıları],[İkinci elemanın katsayıları]]
# b = np.array([12, 10]) # [birinci denklenmin sonucu,ikinci denklemin sonucu]

# print(np.linalg.solve(a, b)) #linalg.solve(birinci denklem, ikinci denklem)
# # [1.85714286 2.71428571]


#############################################
# PANDAS
#############################################

# Pandas Series
# Veri Okuma (Reading Data)
# Veriye Hızlı Bakış (Quick Look at Data)
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
# Apply ve Lambda
# Birleştirme (Join) İşlemleri

#############################################
# Pandas Series
# #############################################
# import pandas as pd

# s= pd.Series([10,77,12,4,5])
# print(s)
# pandas serisi
# 0    10
# 1    77
# 2    12
# 3     4
# 4     5
# print(type(s)) # <class 'pandas.core.series.Series'>
# print(s.index) # RangeIndex(start=0, stop=5, step=1)
# print(s.dtype) # int64 data tipi
# print(s.size) # 5 büyüklük
# print(s.ndim) # 1 boyut bilgisi
# print(s.values) # [10 77 12  4  5]
# print(type(s.values)) # <class 'numpy.ndarray'> indexle ilgilenmediğimiz için numpy array tipindedir
# print(s.head(3)) # ilk 3 ifade
# 0    10
# 1    77
# 2    12
# print(s.tail(3)) # son 3 ifade
# 2    12
# 3     4
# 4     5


#############################################
# Veri Okuma (Reading Data)
#############################################
# import pandas as pd
# df = pd.read_csv('datasets//employee_performance.csv')
# a = df.head()
# print(a)


#############################################
# Veriye Hızlı Bakış (Quick Look at Data)
#############################################
import pandas as pd
import seaborn as sns
import pdb

# df = sns.load_dataset("titanic")
# df.head() # dataframe in ilk verilerine ulaşmak için
# df.tail() # df nin sondan verilerine ulaşmak için
# df.shape  # (891, 15) 891 satır 15 sütun
# df.info() # df hakkında bilgi veriyor
# 
# df.columns # sütunları içeren array i yazar ve dtype ı gösterir
# df.index # RangeIndex(start=0, stop=891, step=1)
# df.describe().T # Verilerimizin sayısal değerlerini,ortalama min max gibi değerleri bize gösterir
# df.isnull().values.any() # values da herhangi virinde isnull ifadesi var mı kontrol etmek için kullanılır. True döner
# df.isnull().sum() # True lar 1 false lar 0 olduğu için isnull değeri olan yani eksiklik bilgisi olan değerlerin toplamını verir 
# df["sex"].head() # belirli bir değerin ilk verilerini verir
# df["sex"].value_counts() #'sex' nesnesinin altında bulunan değerlerin ayrı ayrı toplam değerini verir


#############################################
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
#############################################
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index 
df[0:13] # 0 dan 13 e kadar (13) dahil değil df yi getirir
df.drop(0, axis=0).head() # axis = 0 'satır' anlamına gelir. 0. satırı silmek istiyoruz. 

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10) # delete_indexes ile seçili indexleri silmemizi sağlar

# df = df.drop(delete_indexes, axis=0) # kalıcı olması için df ye tekrar atayabiliriz
# df.drop(delete_indexes, axis=0, inplace=True) # yaptığımız bu değişikliğin kalıcı olması için inplace argumanı kullanılır
#######################
# Değişkeni Indexe Çevirmek
#######################

df["age"].head()
df.age.head()

df.index = df["age"] # yaş bilgisi index olarak atandı

df.drop("age", axis=1).head() # axis = 1 diyerek bu sefer sütunlardan silme işlemi yaptık

df.drop("age", axis=1, inplace=True)
df.head()

#######################
# Indexi Değişkene Çevirmek
#######################

df.index
#df["age"] # 'age' adında bir datamız var mı kontrol ederiz keyerror verir çünkü silmiştik
df["age"] = df.index # ardından indeximizin yerinde bulunan age bilgisini yeni bir sütun oluşturarak 'age' nesnesine atar

df.head()
df.drop("age", axis=1, inplace=True) # age i kalıcı olarak sütundan sildik

df.reset_index().head() # df imiz resetlenmiş oldu
df = df.reset_index() # bunu df e atadık ve kesinleştirmiş olduk
df.head()

#######################
# Değişkenler Üzerinde İşlemler
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None) #... yerine tüm verisetinin kolonlarını görebiliriz
df = sns.load_dataset("titanic")
df.head()

"age" in df # sorgulama amaçlıdır 'True' döner

df["age"].head()  # age kolonunu bize döner
df.age.head() #age kolonunu bize döner

# df["age"].head()
type(df["age"].head()) # <class 'pandas.core.series.Series'> pandas serisidir


df[["age"]].head()
type(df[["age"]].head()) # <class 'pandas.core.frame.DataFrame'> pandas dataframe'idir. çift köşeli parantez kullanmamız gerekir

df[["age", "alive"]] # iki tane birden nesneyi seçebilirz

col_names = ["age", "adult_male", "alive"]
df[col_names] # yukarıdaki 3 adet nesneyi seçmeyi yarar

df["age2"] = df["age"]**2 # dataframe'e 'age2' adında yeni kolon üretir ve 'age' verilerinin karesini alıp yazar
df["age3"] = df["age"] / df["age2"] #aynı şekide 'age1' ve 'age2' nin verilerinin birbirine olan oranını 'age3' e yazar

df.drop("age3", axis=1).head() # age3 nesnesini df'den siler

df.drop(col_names, axis=1).head() #  col_names arrayinde bulunan nesneleri toplu siler

df.loc[:, df.columns.str.contains("age")].head() # Bu ifade dataframein kolonlarındaki 'age' stingini içeren kolonları seçer
df.loc[:, ~df.columns.str.contains("age")].head() # df nin başına ~ ifadesi gelirse bu işlemin tam tersini yapar
#######################
# iloc & loc
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection
df.iloc[0:3] # integer tabanlı seçme 0 dan 3 e kadar 3ü seçmez
df.iloc[0, 0] # x,y ekseninde 0. satır 0. sütun seçme

# loc: label based selection
df.loc[0:3] # label tabanlı da 0 ve 3. index dahil seçim yapar

df.iloc[0:3, 0:3] # bu seçim işleminde integer değerler olmak zorunda 'age' yazarak arayamayız
df.loc[0:3, "age"] # bu şekilde age bloğundaki 0,1,2,3. indexte bulunan değerleri seçebiliriz

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names] # Toplu nesneleri de seçebiliriz


#######################
# Koşullu Seçim (Conditional Selection)
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head() # age nesnesinin 50 den büyük değerlerdeki indexleri getirir
df[df["age"] > 50].count() # yaşı 50 den büyük kaç kişi var

df.loc[df["age"] > 50, ["age", "class"]].head() # yaşın 50 den büyük olduğu yaş ve class nesnelerini birlikte getir

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head() # yaş 50 den büyük ve cinsiyeti erkek olan nesnelerin yaş ve class bilgisini getir

df["embark_town"].value_counts() # embark_town nesnesindeki verilerin ayrı ayrı sayılarını verir

df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]] # yaşı 50den büyük olan Cherbourg'lu y ada Southamptonlu Erkeklerin age, class ve embark_town bilgilerini getir

df_new["embark_town"].value_counts() # yeni df mizde bulunann embark_town bilgisinin ayrı ayrı değerini öğreniriz

#############################################
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
#############################################

# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean() # age nesnesinin verilerinin ortalamasını verir

df.groupby("sex")["age"].mean() # 'sex' nesnesine göre gruplayın ve her grubun yaşa göre ortalamasını yazın

df.groupby("sex").agg({"age": "mean"}) # 'sex' nesnesine göre gruplayın ve her grubun yaşa göre ortalamasını yazın 
                                       # bu kullanımı kullanmamız daha iyi
df.groupby("sex").agg({"age": ["mean", "sum"]}) # yaşın 

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"})
                                #  age  survived   sex
                                # mean      mean count
# sex    embark_town class
# female Cherbourg   First   36.052632  0.976744    43
                #    Second  19.142857  1.000000     7
                #    Third   14.062500  0.652174    23
    #    Queenstown  First   33.000000  1.000000     1
                #    Second  30.000000  1.000000     2
                #    Third   22.850000  0.727273    33
    #    Southampton First   32.704545  0.958333    48
                #    Second  29.719697  0.910448    67
                #    Third   23.223684  0.375000    88
# male   Cherbourg   First   40.111111  0.404762    42
                #    Second  25.937500  0.200000    10
                #    Third   25.016800  0.232558    43
    #    Queenstown  First   44.000000  0.000000     1
                #    Second  57.000000  0.000000     1
                #    Third   28.142857  0.076923    39
    #    Southampton First   41.897188  0.354430    79
                #    Second  30.875889  0.154639    97
                #    Third   26.574766  0.128302   265


#######################
# Pivot table
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked") # survived değişkenini 'sex' ve 'embarked' değerlerine göre analiz eder.
                                              # satırlarda 'sex' değişkenleri male and female
                                              # sütunda embarked değişkenler C, Q ve s vardırr
                                              # kesiştikleri alanda hayatta kalma oranlarının ortalaması bulunmakta
df.pivot_table("survived", "sex", ["embarked", "class"])# sütunda embarked ve class var satırda, sex var sonuçlar survived verileri.

df.head()
df['new_age'] = pd.cut(df['age'], [0,10,18,25,40,90]) # yaş değerini aralıklara bölüp yeni bir nesneye atıyoruz

df.pivot_table('survived','sex',['new_age','class'])

pd.set_option('display.width', 500)

#############################################
# Apply ve Lambda
#############################################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df['age']*2
df['age3'] = df["age"]*5

(df['age']/10).head()
(df["age2"]/10).head()
(df['age3']/10).head()

# for col in df.columns:
#     if 'age' in col:
#         print(col)

for col in df.columns:
    if 'age' in col:
        df[col] = df[col]/10 # eski usüle göre bir fonksiyonu seçtiğimiz kolonlara bu şekilde uygularız.

df[['age','age2','age3']].apply(lambda x : x/10).head() # apply ve lambda yapısı ile burada tek satırda yapabiliriz.


df.loc[:,df.columns.str.contains('age')].apply(lambda x: x/10).head()
df.loc[:,df.columns.str.contains('age')].apply(lambda x: (x-x.mean())/x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

# df.loc[:,['age','age2','age3']]= df.loc[:,df.columns.str.contains('age')].apply(standart_scaler).head()
df.loc[:,df.columns.str.contains('age')] = df.loc[:,df.columns.str.contains('age')].apply(standart_scaler).head() # ya da bu sekilde dışarıda tanımlı bir fonksiyonu apply içerisinde uygulayabiliriz


#############################################
# Birleştirme (Join) İşlemleri (concat)
#############################################
import numpy as np
import pandas as pd
m = np.random.randint(1,30,size=(5,3))
df1 = pd.DataFrame(m, columns=['var1','var2','var3'])
df2 = df1 + 99

pd.concat([df1,df2])

pd.concat([df1,df2],ignore_index=True)


#######################
# Merge ile Birleştirme İşlemleri
#######################
df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})
'''
Ekbilgi: Yukarıdaki DataFrame'de 1-dictionary yapısı var
2-string var, 3-Liste var, 4-integer var. 4 tane farklı veriyapısı bulunan
bunun üzerinden bir dataframe oluşturulmuştur.
'''
# Amaç: Her çalışanın müdürünün bilgisine erişmek istiyoruz.
df3=pd.merge(df1,df2) # Aynı sütunda bulunabilecek verileri kolaylıkla birleştirir.

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})


pd.merge(df3,df4)
#############################################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN
#############################################

#############################################
# MATPLOTLIB
#############################################

# Kategorik değişken: sütun grafik. countplot bar
# Sayısal değişken: histogram (hist), boxplot


#############################################
# Kategorik Değişken Görselleştirme
#############################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df = sns.load_dataset('titanic')
df.head()
df['sex'].value_counts().plot(kind="bar") # value_counts() çok önemli direkt verimizi betimler. plot ile görselleştiririz kind ile türünü belirleriz.
plt.show() # print gibi iş görür bu sefer görüntü ama




#############################################
# Sayısal Değişken Görselleştirme
#############################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"]) # Histogram
plt.show()


plt.boxplot(df['fare']) # Kutu diyagramı
plt.show()


#############################################
# Matplotlib'in Özellikleri
#############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#######################
# plot
#######################

x = np.array([1,8])
y = np.array([0, 150])

plt.plot(x,y)
plt.show()

plt.plot(x,y,'o')
plt.show()

x = np.array([2,4,6,8,10])
y = np.array([1,3,5,7,9])
plt.plot(x,y,'o')
plt.show()


#######################
# marker
#######################

y = np.array([13, 28, 11, 100])
plt.plot(y,marker='o')
plt.show()

markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']

#######################
# line
#######################

y = np.array([13, 28, 11, 100])
plt.plot(y,linestyle='dashdot', color='r')
plt.show()


#######################
# Multiple Lines
#######################

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()


#######################
# Labels
#######################

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)

# Başlık
plt.title('Bu Ana Başlık')
plt.show()

# X eksenine isimlendirme
plt.xlabel('x ekseni isimlendirmesi')
plt.ylabel('y ekseni isimlendirmesi')

plt.grid()

#######################
# Subplots
#######################

# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1,2,1) # 1 satırlık 2 sütünluk grafik oluştur demek en sondaki 1 ise şu an bunun 1. sini oluşturuyorum.
plt.title('1')
plt.plot(x,y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1,2, 2) # 1 satırlık 2 sütünluk grafik oluştur demek en sondaki 1 ise şu an bunun 2. sini oluşturuyorum.
plt.title("2")
plt.title('1')
plt.plot(x, y)
plt.show()


# 3 grafiği bir satır 3 sütun olarak konumlamak.
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

plt.show()

 
#############################################
# SEABORN
#############################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()
df['sex'].value_counts()
sns.countplot(x=df['sex'],data=df) #seaborn ile
plt.show()


df['sex'].value_counts().plot(kind='bar') #matplotlib ile
plt.show()



#############################################
# Sayısal Değişken Görselleştirme
#############################################
sns.boxplot(x=df['total_bill'])
plt.show()


df['total_bill'].hist()
plt.show()



#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################
# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#############################################
# 1. Genel Resim
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head() # ilk 5 satırı getir
df.tail() # son 5 satırı getir
df.shape  # verimizin satır ve sütun sayısını veren tuple
df.info() # colonlar hakkında detaylı bilgi dtype ları non-nullcountlarını verir
df.columns # kolonları getirir
df.index  # index bilgisini getirir
df.describe().T # count       mean        std   min      25%      50%   75%       max her kolonun bu bilgisini verir
df.isnull().values.any() # 0 değeri var mı sorusunu cevaplar
df.isnull().sum() # 0 değeri olan sütunların toplamını verir


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)
df = sns.load_dataset('flights')
check_df(df)


#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["survived"].value_counts()
df["sex"].unique()
df["class"].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]


df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("sdfsdfsdfsdfsdfsd")
    else:
        cat_summary(df, col, plot=True)


df["adult_male"].astype(int)


for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)

def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


cat_summary(df,'adult_male',plot=True)




#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[["age","fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe,numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)


num_summary(df,'age')


for col in num_cols:
    num_summary(df,col)


def num_summary(dataframe,numerical_col,plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df,'age',True)


for col in num_cols:
    num_summary(df,col,plot=True)


#############################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
#################################S############

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()


#docstring
def grab_col_names(dataframe, cat_th=10,  car_th=20):

    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    #cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables:{dataframe.shape[1]}')
    print(f'cat_cols:{len(cat_cols)}')
    print(f'num_cols:{len(num_cols)}')
    print(f'cat_but_car:{len(cat_but_car)}')
    print(f'num_but_cat:{len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)



def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df,col)


def num_summary(dataframe,numerical_col,plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df,col,plot=True)


#bonus
df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes =='bool':
        df[col] = df[col].astype("int64")

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe,numerical_col,plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
    
for col in cat_cols:
    cat_summary(df,col,plot = True)

for col in num_cols:
    num_summary(df,col,plot=True)


#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype('int64')

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_summary(df,'survived')


#######################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
#######################

df.groupby('sex')['survived'].mean()

def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({'TARGET_MEAN':dataframe.groupby(categorical_col)[target].mean()}),end='\n\n\n')

target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df,'survived',col)


#######################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
#######################

df.groupby("survived")["age"].mean()

df.groupby("survived").agg({'age':'mean'})

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_summary_with_num(df,'survived','age')

for col in num_cols:
    target_summary_with_num(df,'survived',col)


#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in ['float64','int64']]

corr = df[num_cols].corr() # korealasyon çıktısı için corr fonksiyonu kullanılır. 
# korealasyon verilerimizin birbiriyle ilişkisini görmemizi sağlar
#-1 ile 1 arasında değer alır
# 1 e ne kadar yakınsa daha şiddetli bir şekilde birlikte hareket edildiğini ifade eder.
# genelde analitik çalışmalarda birbiriyle yüksek korelasyon gösteren verilerden biri silinmek istenir.

sns.set_theme(rc ={'figure.figsize':(12,12)})
sns.heatmap(corr,cmap='RdBu')
plt.show() # korealasyon matrisinin ısı haritasını verir


#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#######################

cor_matrix = df[num_cols].corr().abs()

#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000


#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)


def high_correlated_cols(dataframe,plot= False, corr_th=0.9):
    # Filter only numeric columns
    num_cols = dataframe.select_dtypes(include=[np.number]).columns
    corr = dataframe[num_cols].corr()

    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme(rc={'figure.figsize':(15,15)})
        sns.heatmap(corr,cmap='RdBu')
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df,plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

# Yaklaşık 600 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/train_transaction.csv")
len(df.columns)
df.head()


drop_list = high_correlated_cols(df)
len(df.drop(drop_list, axis=1).columns)


