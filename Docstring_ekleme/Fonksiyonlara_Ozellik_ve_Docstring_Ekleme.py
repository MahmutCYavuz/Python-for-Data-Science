
###############################################
# Fonksiyonlara Özellik Ekleme VE Docstring Ekleme
###############################################

###############################################
# GÖREV 1: Fonksiyonlara Özellik Ekleme
###############################################
# cat_summary() fonksiyonuna 1 özellik ekleyiniz. Bu özellik argumanla biçimlendirilebilir olsun.
# Var olan özelliği de argumandan kontrol edilebilir hale getirebilirsiniz.


# ÖNCESİ
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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# SONRASI
def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


import pandas as pd
df = pd.read_csv("datasets/titanic.csv")
check_df(df, head=3, tail=3, quan=True)


###############################################
# cat_summary() için ratio özelliği argumana verilmiştir.
###############################################

import seaborn as sns
import matplotlib.pyplot as plt

# ÖNCESİ
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Sex")


# SONRASI
def cat_summary(dataframe, col_name, plot=False, ratio=True):
    if ratio:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts()}))
        print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


cat_summary(df, "Sex", plot=False, ratio=False)



###############################################
# GÖREV 2: cat_summary() ve check_df()  fonksiyonuna docstring yazınız.
###############################################

def check_df(dataframe, head=5):
    """

    It gives information about shape of the dataframe, type of its variables, first 5 observations, last 5 observations, total number of missing observations and quartiles.

    Parameters:
    ------
        dataframe : dataframe
            The dataframe whose properties will be defined

        head : int, optional
            Refers to the number of observations that will be displayed from the beginning and end


    Returns:
    ------
        None

    Examples:
    ------
        check_df(df, 7)

    Notes:
    ------
        The number of observations to be viewed from the end is the same as the number of observations to be observed from the beginning.

    """
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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):

    """
        It gives the percentage calculation of the category in the selected variable relative to the entire dataframe.
        Note: The selected variable must be a categorical variable.

        S

        Parameters:
        ------
            dataframe : dataframe
                The dataframe whose percentage of variables will be calculated

            col_name : [str]
                The name of the variable which is calculated as a percentile

            plot : bool, optional
                It decides whether the data results are to be visualized or not.


        Returns:
        ------
            None

        Example:
        ------
            import seaborn as sns
            df = sns.load_dataset("titanic")
            cat_summary(df, "Survived", plot=True)
    """

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


