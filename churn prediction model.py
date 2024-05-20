import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report

#завантаження даних
df = pd.read_csv('your_path')

#розуміння даних
df.head()
df.shape
df.info()
df.columns.values
df.dtypes

#Візуалізація відсутніх значень у вигляді матриці
msno.matrix(df);

#Маніпуляції з даними
df = df.drop(['customerID'], axis = 1)
df.head()

#При глибокому аналізі ми можемо виявити деякі непрямі пропуски в наших даних (які можуть бути у вигляді пробілів)
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()
#Тут ми бачимо, що TotalCharges має 11 відсутніх значень. Давайте перевіримо ці дані.
df[np.isnan(df['TotalCharges'])]
#Також можна помітити, що стовпчик Tenure дорівнює 0 для цих записів, хоча стовпчик MonthlyCharges не порожній. Подивимось, чи є ще якісь інші 0-значення у стовпчику tenure.

df[df['tenure'] == 0].index
#У колонці "tenure" немає додаткових пропущених значень.

#Видалимо рядки з відсутніми значеннями в стовпчиках Tenure, оскільки таких рядків лише 11 і їх видалення не вплине на дані.

df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
df[df['tenure'] == 0].index

#Щоб вирішити проблему відсутності значень у стовпчику TotalCharges, я вирішив заповнити його середнім значенням значень TotalCharges.
df.fillna(df["TotalCharges"].mean())

df.isnull().sum()

df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df.head()

df["InternetService"].describe(include=['object', 'bool'])

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols].describe()
#в даних пристутній баланс між класами churned yes та churned no

#Data Preprocessing
#Splitting the data into train and test sets¶
def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series
  
df = df.apply(lambda x: object_to_int(x))
df.head()

plt.figure(figsize=(14,7))
df.corr()['Churn'].sort_values(ascending = False)

X = df.drop(columns = ['Churn'])
y = df['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40, stratify=y)

def distplot(feature, frame, color='r'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color= color)

num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
for feat in num_cols: distplot(feat, df)

#Оскільки числові характеристики розподілені в різних діапазонах значень, я використаю standard scaler, щоб масштабувати їх до одного діапазону.
df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')),
                       columns=num_cols)
for feat in numerical_cols: distplot(feat, df_std, color='c')

#Розділив стовпчики на 3 категорії: один для стандартизації, один для кодування етикеток і один для гарячого кодування

cat_cols_ohe =['PaymentMethod', 'Contract', 'InternetService'] # ті, що потребують швидкого кодування
cat_cols_le = list(set(X_train.columns)- set(num_cols) - set(cat_cols_ohe)) #ті, що потребують кодування етикеток

scaler= StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


#Побудова моделі
import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

#Обучити модель
lr_model_sm = sm.Logit(y_train, X_train_sm).fit()

#Вивести підсумок
lr_model_sm.summary()

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(roc_auc)




X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

X_test_sm.drop("StreamingMovies", axis='columns')
X_train_sm.drop("StreamingMovies", axis='columns')

#Обучити модель
lr_model_sm = sm.Logit(y_train, X_train_sm).fit()

#Вивести підсумок
lr_model_sm.summary()


X_train = X_train.drop(["StreamingMovies", axis='columns')
X_test = X_test.drop(["StreamingMovies", axis='columns')
X_test = X_test.drop(["gender", "Partner", "Dependents", "MultipleLines", "DeviceProtection", "StreamingTV", "PaymentMethod"], axis='columns')
X_train = X_train.drop(["gender", "Partner", "Dependents", "MultipleLines", "DeviceProtection", "StreamingTV", "PaymentMethod"], axis='columns')
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)
#Після аналізу підсумків видаляємо статистично не значущі показники


#Обучити модель
lr_model_sm = sm.Logit(y_train, X_train_sm).fit()

#Вивести підсумок
lr_model_sm.summary()


lr_pred= lr_model.predict(X_test)
report = classification_report(y_test,lr_pred)
print(report)

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(roc_auc)



from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_VIF(X_train):  # розрахунок VIF для показників
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns # Read the feature names
    vif['VIF'] = [variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])] # розрахунок VIF
    vif['VIF'] = round(vif['VIF'],2)
    vif.sort_values(by='VIF', ascending = False, inplace=True)  
    return(vif) # поверення розрахованного VIFs для всіх показників

calculate_VIF(X_train)
#цей кусок коду для розрахунку VIF запускався періодично після кожної ітерації побудови моделі посля видалення незначущих змінних, щоб видалити корельовані змінні та побудувати точну модель


#приміннення моделі прогнозування відтоку клієнтів на практиці
y_train_pred = lr_model_sm.predict(X_train_sm)
#y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred_final = pd.DataFrame({'Churn':y_train, 'Churn_Prob':y_train_pred})
y_train_pred_final.head()































































































































































































































































































































































































































