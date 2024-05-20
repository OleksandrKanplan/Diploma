#Аналіз даних та обробка даних
import numpy as np
import pandas as pd
from collections import Counter

#Вiзуалiзацiя
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
%matplotlib inline

# Стилi графiкiв
sns.set_context("paper")
style.use('fivethirtyeight')

# Machine Learning бiблiотеки

#Sci-kit learn бiблiотеки
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

#бібліотеки statmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


lead = pd.read_csv("file_path")
lead.head()

#розмір бази даних
print("Database dimension     :",lead.shape)
print("Database size          :",lead.size)
print("Number of Row          :",len(lead.index))
print("Number of Columns      :",len(lead.columns))

#перевірка статистики числових стовпців
lead.describe()

#інформацію про типи стовпців тощо. 
lead.info()


lead = lead.replace('Select', np.nan)
plt.figure(figsize = (18,8))
sns.heatmap(lead.isnull(),cbar = False)
plt.show()


#Нульові значення по стовпчиках у наборі даних поїздів 
null_perc = pd.DataFrame(round((lead.isnull().sum())*100/lead.shape[0],2)).reset_index()
null_perc.columns = ['Column Name', 'Null Values Percentage']
null_value = pd.DataFrame(lead.isnull().sum()).reset_index()
null_value.columns = ['Column Name', 'Null Values']
null_lead = pd.merge(null_value, null_perc, on='Column Name')
null_lead.sort_values("Null Values", ascending = False)

#побудова графіка відсотка нульових значень
sns.set_style("white")
fig = plt.figure(figsize=(12,5))
null_lead = pd.DataFrame((lead.isnull().sum())*100/lead.shape[0]).reset_index()
ax = sns.pointplot("index",0,data=null_lead)
plt.xticks(rotation =90,fontsize =9)
ax.axhline(45, ls='--',color='red')
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.xlabel("COLUMNS")
plt.show()


Row_Null50_Count = len(lead[lead.isnull().sum(axis=1)/lead.shape[1]>0.5])
print( 'Total number of rows with more than 50% null values are : ', Row_Null50_Count)

#Є 17 стовпчиків з нульовими значеннями. 7 стовпців мають більше 45% невідомих, які ми повинні відкинути, оскільки імплементація цих стовпців призведе до зміщення. Немає рядків, які мають більше 50% нульових значень.


print("Total number of duplicate values in Prospect ID column :" , lead.duplicated(subset = 'Prospect ID').sum())
print("Total number of duplicate values in Lead Number column :" , lead.duplicated(subset = 'Lead Number').sum())

#Ідентифікатор потенційного клієнта, і номер ліда є унікальними стовпчиками, і тому нам не знадобляться для прогнозування


#DATA CLEANING

#Очевидно, що ідентифікаційний номер потенційного клієнта та номер ліда - це дві змінні, які лише вказують на ідентифікаційний номер контактної особи, і їх можна вилучити. Ми також видалимо стовпці, які мають більше 45% нульових значень.
#Крім того, деякі змінні, такі як "Якість лідів", "Теги", "Оцінка асиметрії", "Профіль" тощо, створюються відділом продажів після того, як вони зв'яжуться з потенційним лідом. Ці змінні не будуть доступні для побудови моделі, оскільки ці характеристики не будуть доступні до контакту з потенційним клієнтом.
#Остання помітна активність - це проміжний стовпчик, який є оновленням інформації під час контакту представника відділу продажів з потенційним клієнтом.
#Таким чином, ми також можемо опустити ці стовпці.

cols_to_drop = ['Prospect ID','Lead Number','How did you hear about X Education','Lead Profile',
                'Lead Quality','Asymmetrique Profile Score','Asymmetrique Activity Score',
               'Asymmetrique Activity Index','Asymmetrique Profile Index','Tags','Last Notable Activity']

#видалення непотрібних стовпців

lead.drop(cols_to_drop, 1, inplace = True)
len(lead.columns)

#Ми успішно видалили 10 стовпців, де стовпці мали високі нульові значення або містили інформацію, яка не буде доступна для моделей під час запуску, оскільки вони розраховуються/вибираються торговим персоналом під час контакту з потенційними клієнтами.

#Розмежування категоріальних і числових значень
categorical_col = lead.select_dtypes(exclude =["number"]).columns.values
numerical_col = lead.select_dtypes(include =["number"]).columns.values
print("CATEGORICAL FEATURES : \n {} \n\n".format(categorical_col))
print("NUMERICAL FEATURES : \n {} ".format(numerical_col))

#Обробка нульового значення категорійних стовпців
def Cat_info(df, categorical_column):
    df_result = pd.DataFrame(columns=["columns","values","unique_values","null_values","null_percent"])
    
    df_temp=pd.DataFrame()
    for value in categorical_column:
        df_temp["columns"] = [value]
        df_temp["values"] = [df[value].unique()]
        df_temp["unique_values"] = df[value].nunique()
        df_temp["null_values"] = df[value].isna().sum()
        df_temp["null_percent"] = (df[value].isna().sum()/len(df)*100).round(1)
        df_result = df_result.append(df_temp)
    
    df_result.sort_values("null_values", ascending =False, inplace=True)
    df_result.set_index("columns", inplace=True)
    return df_result

df_cat = Cat_info(lead, categorical_col)
df_cat

#Деякі стовпці мають лише 1 категорію, наприклад, "Журнал", "Згоден сплатити суму чеком" тощо. Ці стовпці не додають ніякої цінності до моделі і можуть бути видалені.
#Деякі стовпці мають одне із значень "Вибрати". Їх слід розглядати як нульові значення. Для цих стовпців потрібно оновити значення даних

#Додавання стовпців до col_to_drop, де присутнє лише 1 значення категорії
cols_to_drop = df_cat[df_cat['unique_values']==1].index.values.tolist() 
cols_to_drop

lead.drop(cols_to_drop, 1, inplace = True)
len(lead.columns)

categorical_col = lead.select_dtypes(exclude =["number"]).columns.values
new_cat = Cat_info(lead, categorical_col)
new_cat

lead['City'].value_counts(normalize=True)*100
lead.groupby(['Country','City'])['Country'].count()
style.use('fivethirtyeight')
ax = sns.countplot(lead['City'],palette = 'Set2')
plt.xticks(rotation = 90)
plt.show()

#Оскільки майже 40% значень є невідомими, ми не можемо робити імплікації за допомогою моди, оскільки це може призвести до викривлення всіх даних. Крім того, X-Education - це платформа для онлайн-навчання. Інформація про місто не буде дуже корисною, оскільки потенційні студенти можуть отримати доступ до будь-яких курсів онлайн, незважаючи на їхнє місто. Ми вилучимо цей стовпчик з аналізу.

lead.drop("City",axis=1, inplace = True)
len(lead.columns)

lead['Specialization'].value_counts(normalize = True)*100
plt.figure(figsize=(12,6))
ax = sns.countplot(lead['Specialization'],palette = 'Set2')
plt.xticks(rotation = 90)
plt.show()

#Можливо, що керівник не має спеціалізації або є студентом і ще не має досвіду роботи, тому не ввів жодного значення. Ми створимо нову категорію під назвою "Інші", щоб замінити нульові значення.

lead['Specialization'] = lead['Specialization'].replace(np.nan, 'Others')
plt.figure(figsize=(12,6))
ax = sns.countplot(lead['Specialization'],palette = 'Set2')
plt.xticks(rotation = 90)
plt.show()

lead['What matters most to you in choosing a course'].value_counts(normalize = True)*100
#Що для вас найважливіше при виборі курсу. Оскільки дані викривлені, ми можемо видалити стовпчик.
lead.drop('What matters most to you in choosing a course', axis = 1, inplace=True)
len(lead.columns)

#Яка ваша зайнятicть?
lead['What is your current occupation'].value_counts(normalize=True)*100


#85,5% значень - це "Безробітні". Якщо ми інтерполюємо дані як "Безробітний", то дані стануть більш викривленими. Таким чином, ми інтерполюємо значення як "Невідомо".

lead['What is your current occupation'] = lead['What is your current occupation'].replace(np.nan, 'Unknown')
lead['What is your current occupation'].value_counts(normalize = True)*100

#Давайте перевіримо, як розподіляються дані про країну
lead['Country'].value_counts(normalize=True)

#Дані про країну сильно викривлені, оскільки 95% даних віднесено до Індії. Як і для міста, дані про країну не потрібні для побудови моделі, оскільки X-Education є онлайн-платформою. Ми також опустимо колонки з країнами.
lead.drop('Country', axis = 1, inplace = True)
len(lead.columns)

#Остання дiяльнiсть
print("Number of null values in Last Activity column is : ", lead['Last Activity'].isnull().sum())
print("Percentage of null values in Last Activity column is : ", round(lead['Last Activity'].isnull().sum()/lead.shape[0]*100,2))
lead['Last Activity'].value_counts(normalize = True)*100
#Оскільки ми не знаємо, що може бути останньою дією, ми замінимо її на найчастішу дію "Відкрито імейл".

lead['Last Activity'] = lead['Last Activity'].replace(np.nan, 'Email Opened')
print("Number of null values in Last Activity column is : ", lead['Last Activity'].isnull().sum())

#Lead джерело
print("Number of null values in Lead Source column is : ", lead['Lead Source'].isnull().sum())
print("Percentage of null values in Lead Source column is : ", round(lead['Lead Source'].isnull().sum()/lead.shape[0]*100,2))
lead['Lead Source'].value_counts(normalize = True)*100
#Оскільки Google є найбільш часто використовуваним джерелом, ми замінимо нульові значення на Google. Існує категорія "google", яка збігається з "Google". Ми замінимо значення

lead['Lead Source'] = lead['Lead Source'].replace(np.nan, 'Google')
lead['Lead Source'] = lead['Lead Source'].replace(['google'], 'Google')
print("Number of null values in Lead Source column is : ", lead['Lead Source'].isnull().sum())

#Ми успішно проімпліцитували всі категоріальні стовпці. Тепер давайте розглянемо числові стовпці.

#Перевірка унікальних значень і нульових значень для категорійних стовпців
def Num_info(df, numeric_column):
    df_result = pd.DataFrame(columns=["columns","null_values","null_percent"])
    
    df_temp=pd.DataFrame()
    for value in numeric_column:
        df_temp["columns"] = [value]
        df_temp["null_values"] = df[value].isna().sum()
        df_temp["null_percent"] = (df[value].isna().sum()/len(df)*100).round(1)
        df_result = df_result.append(df_temp)
    
    df_result.sort_values("null_values", ascending =False, inplace=True)
    df_result.set_index("columns", inplace=True)
    return df_result

df_num = Num_info(lead,numerical_col)
df_num

#Загальна кількість відвідувань 
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
sns.distplot(lead['TotalVisits'])
plt.subplot(1,2,2)
sns.boxplot(lead['TotalVisits'])
plt.show()
#Оскільки ми бачимо, що в даних є деякі викиди, ми будемо імпліцирувати медіану, а не середнє значення.

lead['TotalVisits'].fillna(lead['TotalVisits'].median(), inplace=True)
lead['TotalVisits'].isnull().sum()

#Перегляд сторінок за візит
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
sns.distplot(lead['Page Views Per Visit'])
plt.subplot(1,2,2)
sns.boxplot(lead['Page Views Per Visit'])
plt.show()

#Оскільки ми бачимо, що в даних є деякі викиди, ми будемо імпліцирувати медіану, а не середнє значення.
lead['Page Views Per Visit'].fillna(lead['Page Views Per Visit'].median(), inplace=True)
lead['Page Views Per Visit'].isnull().sum()


#Data Imbalance

converted = lead['Converted'].value_counts().rename_axis('unique_values').to_frame('counts')
converted
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(converted.counts, labels = ['No','Yes'],colors = ['red','green'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

#У співвідношенні конверсії лідів 38,5% перетворилися на ліди, тоді як 61,5% не перетворилися на ліди. Таким чином, це здається збалансованим набором даних.

#Підготовка даних
lead.describe(percentiles=[.1,.5,.25,.75,.90,.95,.99])
numerical_col
#Побудова числових стовпчиків для значень викидів
i=1
plt.figure(figsize=[16,8])
for col in numerical_col:
    plt.subplot(2,2,i)
    sns.boxplot(y=lead[col])
    plt.title(col)
    plt.ylabel('')
    i+=1

#Хоча викиди в TotalVisits і Page Views per Visit показують дійсні значення, це може призвести до неправильної класифікації результатів і, як наслідок, створити проблеми, коли ви робите висновки за неправильною моделлю. Логістична регресія сильно залежить від пропусків. Тому давайте обмежимо значення TotalVisits і Page Views per Visit до 95-го процентиля з наступних причин:
#Набір даних досить великий
#95-й процентиль і 99-й процентиль цих стовпчиків дуже близькі, а отже, вплив обмеження до 95-го або 99-го процентиля буде однаковим

#Обмеження даних 95% перцентильним значенням
Q4 = lead['TotalVisits'].quantile(0.95) # Отримаймо 95-й квантиль
print("Total number of rows getting capped for TotalVisits column : ",len(lead[lead['TotalVisits'] >= Q4]))
lead.loc[lead['TotalVisits'] >= Q4, 'TotalVisits'] = Q4 # обмеження викидів

Q4 = lead['Page Views Per Visit'].quantile(0.95) # Отримаймо 95-й квантиль
print("Total number of rows getting capped for Page Views Per Visit column : ",len(lead[lead['Page Views Per Visit'] >= Q4]))
lead.loc[lead['Page Views Per Visit'] >= Q4, 'Page Views Per Visit'] = Q4 # обмеження викидів

lead.describe(percentiles=[.1,.5,.25,.75,.90,.95,.99])



#Перетворення бінарних категорій
lead.nunique().sort_values()
#Перевірка кількості унікальних значень для категорійних стовпців

#Перевірка категорійних значень для функції "Не відкривав email"
lead['Do Not Email'].value_counts()


#Список змінних для мапування

#Визначення функції мапування
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

#Застосування функції до головного списку змінних YES/NO
lead['Do Not Email'] = lead[['Do Not Email']].apply(binary_map)

#повторна перевірка категорійних значень для функції "Не відкривав email"
lead['Do Not Email'].value_counts()


#Фіктивні змінні
#Створення фіктивної змінної для деяких категоріальних змінних і відкидання першої з них.
dummy1 = pd.get_dummies(lead[['Lead Origin', 'Lead Source', 'Occupation', 'Last Activity', 'Specialization']], drop_first=True)

#Додавання результатів до основного фрейму даних
lead = pd.concat([lead, dummy1], axis=1)

lead.head()


# Ми створили фіктивні змінні для наведених нижче змінних, тому ми можемо їх опустити
lead = lead.drop(['Lead Origin', 'Lead Source', 'Occupation', 'Last Activity', 'Specialization'], axis=1)
lead.info()



corr_lead = lead.corr()
corr_lead = corr_lead.where(np.triu(np.ones(corr_lead.shape),k=1).astype(np.bool))
corr_df = corr_lead.unstack().reset_index()
corr_df.columns =['VAR1','VAR2','Correlation']
corr_df.dropna(subset = ["Correlation"], inplace = True) 
corr_df.sort_values(by='Correlation', ascending=False, inplace=True)

#Топ-5 позитивно корельованих змінних
corr_df.head(5)

corr_df.sort_values(by='Correlation', ascending=True, inplace=True)

#Топ-5 негативно корельованих змінних
corr_df.head(5)

#Train - Test Split
# цільова змінна
Y = lead['Converted']
X = lead.drop(['Converted'], axis=1)

# Поділ даних на навчальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)

#Перевірка форми створених тренувальних та тестових DF
print(" Shape of X_train is : ",X_train.shape)
print(" Shape of y_train is : ",y_train.shape)
print(" Shape of X_test is  : ",X_test.shape)
print(" Shape of y_test is  : ",y_test.shape)

#Масштабування показників
#Ми стандартизували числові атрибути, щоб вони мали спільне середнє значення, рівне нулю, якщо вони були виміряні за різними шкалами
scaler = StandardScaler()

X_train[['Total Time Spent on Website','Page Views Per Visit','TotalVisits']] = scaler.fit_transform(X_train[['Total Time Spent on Website','Page Views Per Visit','TotalVisits']])
X_train.head()
#Тепер, коли наші датассети Train і Test готові, а дані в Train стандартизована, давайте спробуємо побудувати деяку модель за допомогою логістичної регресії.

#Побудова моделі
#Використання RFE для зменшення кількості функцій з 54 до 20
logreg = LogisticRegression()
rfe = RFE(logreg, 20)           
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns, rfe.support_, rfe.ranking_))

#перевірка того, які стовпці залишилися після RFE
rfe_col = X_train.columns[rfe.support_]
rfe_col

#Стовпці, які були видалені після RFE
X_train.columns[~rfe.support_]

#Функції для багаторазового повторення регресійної моделі Logictis та розрахунку VIF

#функція для побудови логістичної регресійної моделі
def build_logistic_model(feature_list):
    X_train_local = X_train[feature_list] #отримати список функцій для VIF
    X_train_sm = sm.add_constant(X_train_local) #обов'язково бібліотекою statsmodels   
    log_model = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial()).fit() #побудувати модель і дізнатися коефіцієнти  
    return(log_model, X_train_sm) #повертає модель та X_train з константою 

#функція для розрахунку VIF
def calculate_VIF(X_train):  # Розрахувати VIF для показників
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns # Читання назви показників
    vif['VIF'] = [variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])] # розрахунок VIF
    vif['VIF'] = round(vif['VIF'],2)
    vif.sort_values(by='VIF', ascending = False, inplace=True)  
    return(vif) # повертає обчислене значення VIF для всіх показників

features = list(rfe_col) #Використовуйте вибрані змінні RFE
log_model1, X_train_sm1 = build_logistic_model(features) #Виклик функції та отримання моделі X_train_sm для прогнозування
log_model1.summary()
calculate_VIF(X_train)

features.remove('Occupation_Housewife')
#Після видалення статистично не значущих показників побудуємо модель ще раз

log_model2, X_train_sm2 = build_logistic_model(features) #Виклик функції та отримання моделі X_train_sm для прогнозування
log_model2.summary()
calculate_VIF(X_train[features])

#Після видалення статистично не значущих показників побудуємо модель ще раз
features.remove('Specialization_Retail Management')


log_model3, X_train_sm3 = build_logistic_model(features) #Виклик функції та отримання моделі X_train_sm для прогнозування
log_model3.summary()
calculate_VIF(X_train[features])

#Після видалення статистично не значущих показників побудуємо модель ще раз
features.remove('Lead Source_Facebook')

log_model4, X_train_sm4 = build_logistic_model(features) #Виклик функції та отримання моделі X_train_sm для прогнозування
log_model4.summary()
calculate_VIF(X_train[features])

#Після видалення статистично не значущих показників побудуємо модель ще раз
features.remove('Specialization_Rural and Agribusiness')


log_model5, X_train_sm5 = build_logistic_model(features) #Виклик функції та отримання моделі X_train_sm для прогнозування
log_model5.summary()
calculate_VIF(X_train[features])

#Всі показникові змінні мають значні P-значення та VIF менше 0.05. Тому далі ми розглянемо значення WoE та коефіцієнта для вилучення параметрів. Для коефіцієнта ми сконцентруємося на від'ємних коефіцієнтах, які потрібно видалити, оскільки ми хочемо мати більше позитивних ознак, які можуть вказувати на визначення правильного кандидата для конверсії лідів або на те, як покращити подальшу роботу з лідами.

def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    
    dset = dset.sort_values(by='WoE')
    
    return dset, iv

for col in lead.columns:
    if col in features:
        df, iv = calculate_woe_iv(lead, col, 'Converted')
        print('IV score of column : ',col, " is ", round(iv,4))

 #Ми вилучимо ознаку "Професія_Невідома" через високий від'ємний коефіцієнт. Крім того, Occupation_Unknown є імпліцитними даними шляхом обробки нульових значень. Це означає, що ця ознака вказує на те, що деякі потенційні клієнти не заповнили цю колонку "Рід занять". Цю ознаку важко інтерпретувати та вживати заходів щодо неї в майбутньому. Тому ми спочатку видалимо цю ознаку.
features.remove('Occupation_Unknown')
log_model6, X_train_sm6 = build_logistic_model(features) #Виклик функції та отримання моделі X_train_sm для прогнозування
log_model6.summary()
calculate_VIF(X_train[features])

for col in lead.columns:
    if col in features:
        df, iv = calculate_woe_iv(lead, col, 'Converted')
        print('IV score of column : ',col, " is ", round(iv,4))

#Ми вилучимо "Спеціалізація_Інші" через складність інтерпретації даних, оскільки ця категорія "Інші" є поєднанням різних спеціалізацій, які були об'єднані в менші частини.

features.remove('Specialization_Others')
log_model7, X_train_sm7 = build_logistic_model(features) #Виклик функції та отримання моделі X_train_sm для прогнозування
log_model7.summary()
calculate_VIF(X_train[features])


for col in lead.columns:
    if col in features:
        df, iv = calculate_woe_iv(lead, col, 'Converted')
        print('IV score of column : ',col, " is ", round(iv,4))

#Ми видалимо "Спеціалізацію_Готельний бізнес", оскільки вона має найнижче значення WoE, а також від'ємне значення коефіцієнту.
features.remove('Specialization_Hospitality Management')
log_model8, X_train_sm8 = build_logistic_model(features) #Виклик функції та отримання моделі X_train_sm для прогнозування
log_model8.summary()
calculate_VIF(X_train[features])

#Ми видалимо "Остання активність_Інша активність", оскільки це параметр, створений шляхом об'єднання декількох менших категорій останньої активності, і % цих даних у всій базі даних становить менше 0,03%.

features.remove('Last Activity_Other Activity')
log_model9, X_train_sm9 = build_logistic_model(features) #Виклик функції та отримання моделі X_train_sm для прогнозування
log_model9.summary()
len(features)
#Стабільною моделлю є модель №9, де всі P-значення ознак є значущими, а значення VIF є меншими за 3, що свідчить про незначну мультиколінеарність. Більшість коефіцієнтів ознак є додатними. Ми оберемо модель №8 як остаточну модель і оцінимо її на навчальній та тестовій вибірках даних.


#Створіть матрицю для друку точності, чутливості та специфічності
def lg_metrics(confusion_matrix):
    TN =confusion_matrix[0,0]
    TP =confusion_matrix[1,1]
    FP =confusion_matrix[0,1]
    FN =confusion_matrix[1,0]
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    speci = TN/(TN+FP)
    sensi = TP/(TP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)
    FPR = FP/(TN + FP)
    FNR = FN/(TP + FN)
    pos_pred_val = TP /(TP+FP)
    neg_pred_val = TN /(TN+FN)
    
    print ("Model Accuracy value is              : ", round(accuracy*100,2),"%")
    print ("Model Sensitivity value is           : ", round(sensi*100,2),"%")
    print ("Model Specificity value is           : ", round(speci*100,2),"%")
    print ("Model Precision value is             : ", round(precision*100,2),"%")
    print ("Model Recall value is                : ", round(recall*100,2),"%")
    print ("Model True Positive Rate (TPR)       : ", round(TPR*100,2),"%")
    print ("Model False Positive Rate (FPR)      : ", round(FPR*100,2),"%")
    print ("Model Poitive Prediction Value is    : ", round(pos_pred_val*100,2),"%")
    print ("Model Negative Prediction value is   : ", round(neg_pred_val*100,2),"%")


#Отримання прогнозованих значень на навчальній вибірці
y_train_pred = log_model9.predict(X_train_sm9)
y_train_pred[:10]

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]

#Створення фрейму даних з фактичним прапором Converted та ймовірностями Predicted
y_train_pred_final = pd.DataFrame({'Converted_IND':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['Prospect_IND'] = y_train.index
y_train_pred_final.head()

#Пошук оптимальної точки відсікання
# Створимо стовпці з різними ймовірнісними відсіченнями 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# Тепер розрахуємо чутливість та специфічність точності для різних імовірнісних відсікань.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci','Precision','Recall'])
from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final['Converted_IND'], y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    prec, rec, thresholds = precision_recall_curve(y_train_pred_final['Converted_IND'], y_train_pred_final[i])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci, prec[1], rec[1]]
cutoff_df



#Побудуємо графіки чутливості та специфічності точності для різних ймовірностей.
plt.figure(figsize=(18,8))
sns.set_style("whitegrid")
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.xticks(np.arange(0,1,step=0.05),size=8)
plt.axvline(x=0.335, color='r', linestyle='--') # додавання осьової лінії
plt.yticks(size=12)
plt.show()
#З наведеного графіка видно, що 0,335 є ідеальною точкою відсікання

y_train_pred_final['final_predicted_1'] = y_train_pred_final['Converted_Prob'].map( lambda x: 1 if x > 0.335 else 0)
y_train_pred_final.drop([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],axis = 1, inplace = True) # deleting the unnecessary columns
y_train_pred_final.head()


#Призначимо Lead_score для лідів у наборі даних Train Data Set
y_train_pred_final['lead_score_1']=(y_train_pred_final['Converted_Prob']*100).astype("int64")
y_train_pred_final.sort_values(by='Converted_Prob',ascending=False)


#Точність, відгук і результат F1
print( metrics.classification_report( y_train_pred_final['Converted_IND'], y_train_pred_final['final_predicted_1'] ) )

#Запис значень FPR, TPR та порогових значень:
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final['Converted_IND'], y_train_pred_final['Converted_Prob'] , drop_intermediate = False )

#побудова кривої ROC 
draw_roc(y_train_pred_final['Converted_IND'], y_train_pred_final['Converted_Prob'])

#Кут нахилу ROC-кривої дорівнює 0,88, що свідчить про те, що модель є хорошою.

