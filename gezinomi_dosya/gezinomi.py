import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', '{:.2f}'.format)

# Soru1 : miuul_gezinomi.xlsx dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_excel("gezinomi_dosya\\miuul_gezinomi.xlsx")
def check_df(dataframe, head):
    print(f"İlk {head} Gözlem Birimi")
    print(dataframe.head(head))

    print("Veri Seti Boyut Bilgisi")
    print(dataframe.shape)

    print("Değişken İismleri")
    print(dataframe.columns)

    print("Eksik Değer Kontrolü")
    print(dataframe.isnull().sum())

    print("Veri Seti Hakkında Genel Bilgiler")
    print(dataframe.info())

check_df(dataframe=df, head=10)
# Soru 2: Kaç unique şehir vardır? Frekansları nedir? 
df['SaleCityName'].nunique()
df['SaleCityName'].value_counts()

# Soru 3:Kaç unique Concept vardır?
df['ConceptName'].nunique()

# Soru4: Hangi Concept’den kaçar tane satış gerçekleşmiş?
df['ConceptName'].value_counts()

# Soru5: Şehirlere göre satışlardan toplam ne kadar kazanılmış?
df.groupby('SaleCityName').agg({'Price':'sum'})

# Soru6:Concept türlerine göre göre ne kadar kazanılmış?
df.groupby('ConceptName').agg({'Price':'sum'})

# Soru7: Şehirlere göre PRICE ortalamaları nedir?
df.groupby('SaleCityName').agg({'Price':'mean'})

# Soru 8:Conceptlere göre PRICE ortalamaları nedir?
df.groupby('ConceptName').agg({'Price':'mean'})

#Soru 9: Şehir-Concept kırılımında PRICE ortalamalarınedir?
df.pivot_table('Price','SaleCityName','ConceptName')
df.groupby(['ConceptName','SaleCityName']).agg({'Price':'mean'})

############################################################
# Görev 2: SaleCheckInDayDiff değişkenini kategorik bir değişkene çeviriniz.
############################################################
# • SaleCheckInDayDiff değişkeni müşterinin CheckIn tarihinden ne kadar önce satin alımını tamamladığını gösterir.
# • Aralıkları ikna edici şekilde oluşturunuz.
# Örneğin: ‘0_7’, ‘7_30', ‘30_90', ‘90_max’ aralıklarını kullanabilirsiniz.
# • Bu aralıklar için "Last Minuters", "Potential Planners", "Planners", "Early Bookers“ isimlerini kullanabilirsiniz.
bins = [-1,7,30,90,df['SaleCheckInDayDiff'].max()]
labels = ['Last Minuters', 'Potential Planners', 'Planners', 'Early Bookers']

df['EB_Score'] = pd.cut(df['SaleCheckInDayDiff'], bins, labels=labels)
df.head(50).to_excel('ebscorew.xlsx',index=False)

############################################################
# Görev 3: Şehir-Concept-EB Score, Şehir-Concept- Sezon, Şehir-Concept-CInDay kırılımında ortalama ödenen ücret ve yapılan işlem sayısı cinsinden
# inceleyiniz ?
############################################################

#Şehir-Concept-EB Score
df.groupby(by=['SaleCityName','ConceptName','EB_Score']).agg({'Price':['mean','count']})

#Şehir-Concept- Sezon
df.groupby(by=['SaleCityName','ConceptName','Seasons']).agg({'Price':['mean','count']})

#Şehir-Concept-CInDay
df.groupby(by=['SaleCityName','ConceptName','CInDay']).agg({'Price':['mean','count']})


############################################################
# 4: City-Concept-Season kırılımının çıktısını PRICE'a göre sıralayınız.
############################################################
agg_df = df.groupby(by=['SaleCityName','ConceptName','Seasons']).agg({'Price':'mean'}).sort_values(by='Price',ascending=False)
agg_df.head(20)


############################################################
# Görev 5: Indekste yer alan isimleri değişken ismine çeviriniz.
############################################################
agg_df=agg_df.reset_index() 
# reset index sayesinde her price değerimiz tek sayırda daha düzenli bir görünüm elde etti.


############################################################
# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
############################################################
# • Yeni seviye tabanlı satışları tanımlayınız ve veri setine değişken olarak ekleyiniz.
# • Yeni eklenecek değişkenin adı: sales_level_based
# • Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek sales_level_based değişkenini oluşturmanız gerekmektedir.
############################################################

agg_df['sales_level_based'] = (agg_df['SaleCityName']+'_'+  agg_df['ConceptName']+'_'+agg_df['Seasons']).str.upper()
agg_df.head(15)

############################################################
# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
############################################################
# • Yeni personaları PRICE’a göre 4 segmente ayırınız.
# • Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# • Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
############################################################

agg_df['SEGMENT']=pd.qcut(agg_df['Price'], 4 ,labels = ['D','C','B','A'])
agg_df.head(15)
agg_df.groupby(by=['SEGMENT']).agg({'Price':['mean','max','sum']},).sort_values(by='SEGMENT',ascending=False)


############################################################
# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
############################################################
# • Antalya’da herşey dahil ve yüksek sezonda tatil yapmak isteyen bir kişinin ortalama ne kadar gelir kazandırması beklenir?
# • Girne’de yarım pansiyon bir otele düşük sezonda giden bir tatilci hangi segmentte yer alacaktır?
############################################################


new_user= agg_df[agg_df['sales_level_based'] == 'ANTALYA_HERŞEY DAHIL_HIGH']
new_user['Price'].mean()


'GIRNE_YARIM PANSIYON_LOW'
agg_df[agg_df['sales_level_based']=='GIRNE_YARIM PANSIYON_LOW']


