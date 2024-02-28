import numpy as np
import pandas as pd
import seaborn as sns


#Görev 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük
#harfe çeviriniz ve başına NUM ekleyiniz.
df = sns.load_dataset('car_crashes')
['NUM_'+col.upper() if df[col].dtype != 'O' else col.upper()for col in df.columns]



# Görev 2: List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan
# değişkenlerin isimlerinin sonuna "FLAG" yazınız.

[col.upper()+'_FLAG' if 'no' not in col else col.upper()for col in df.columns]


#Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan
# değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

og_list =['abbrev','no_previous']
new_cols =[col for col in df.columns if col not in og_list]
# print(new_cols)
new_df = df.head()
# print(new_df)

####################################
# Pandas Alıştırmaları
####################################
import numpy as np
import pandas as pd
import seaborn as sns
import pdb
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
#Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
df = sns.load_dataset('titanic')
#Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
df['sex'].value_counts()
#Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
df.nunique()
#Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.
df['pclass'].unique()
#Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
cols = ['pclass','parch']
df[cols].nunique()
#Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.

df['embarked'].dtype
df["embarked"] = df["embarked"].astype("category")

# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df['embarked'] == 'C'].head()

# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df['embarked'] != 'S'].head()

# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df[(df["age"] < 30) & (df['sex']=='female')].head()

# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
df[(df["fare"] > 500) | (df['age']>70)].head()

# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()

# Görev 12: who değişkenini dataframe’den çıkarınız.
df.drop('who',axis = 1).head()

# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
df['deck'].mode()[0]
df['deck']=df['deck'].fillna(df['deck'].mode()[0])

# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df['age']=df['age'].fillna(df['age'].median())
df['age'].isnull().sum()
# pdb.set_trace()

# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(['pclass','sex']).agg({'survived':['sum','count','mean']})

# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
# setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)

df['age_flag']= df['age'].apply(lambda x:1 if x <30 else 0).head()
df['age_flag'].head()


# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
df = sns.load_dataset('tips')


# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
df.groupby('time').agg({'total_bill':['sum','min','max','mean']})


# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
df.groupby(['day','time']).agg({'total_bill':['sum','min','max','mean']})

# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
df[(df['time']=='Launch') & (df['sex']=='Female')].groupby('day').agg({'total_bill':['sum','min','max','mean'],'tip':['sum','min','max','mean']})

# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
df.loc[(df['size']<3) & (df['total_bill']>10),'total_bill'].mean()

# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
df['total_bill_tip_sum']= df['total_bill'] + df['tip']

# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
df2 = df.sort_values('total_bill_tip_sum',ascending=False)[:30]
pdb.set_trace()
