import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

titanic_df = pd.read_csv("train (1).csv")

titanic_df["Sex"].replace(to_replace="male",value=1,inplace=True)
titanic_df["Sex"].replace(to_replace="female",value=0,inplace=True)

dummie= pd.get_dummies(titanic_df["Embarked"])
titanic_df = pd.concat([titanic_df,dummie],axis=1)
titanic_df.drop(["Embarked"], inplace=True, axis=1)

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df["C"]=titanic_df["C"].astype(np.int64)
titanic_df["Q"]=titanic_df["Q"].astype(np.int64)
titanic_df["S"]=titanic_df["S"].astype(np.int64)
titanic_df["Fare"]=titanic_df["Fare"].astype(np.int64)
titanic_df["Age"]=titanic_df["Age"].astype(int)

X = titanic_df[["Pclass","Sex","Age","SibSp","Parch","Fare","C","Q","S"]]
y = titanic_df["Survived"]

X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print("Tüm veri kümesi '0' yüzdesi : %{:.0f} ".format(len(y[y==0])/len(y)*100))
print("Test verisi '0' yüzdesi     : %{:.0f} ".format(len(y_test[y_test==0])/len(y_test)*100))
print("Eğitim verisi '0' yüzdesi   : %{:.0f} ".format(len(y_egitim[y_egitim==0])/len(y_egitim)*100))


model = LogisticRegression()
model.fit(X_egitim, y_egitim)
tahmin_eğitim = model.predict(X_egitim)
tahmin_test = model.predict(X_test)
print("İlk modelimizin score'u:", model.score(X_test, y_test))
print()

lrm = LogisticRegression()
cv = cross_validate(estimator=lrm,
                     X=X,
                     y=y,
                     cv=10,
                     return_train_score=True
                    )
print("Cross validation:")
print('Test Skorları            : ', cv['test_score'], sep = '\n')
print("-"*50)
print('Eğitim Skorları          : ', cv['train_score'], sep = '\n')
print()
print('Test Kümesi   Ortalaması : ', cv['test_score'].mean())
print('Eğitim Kümesi Ortalaması : ', cv['train_score'].mean())
print()

cv = cross_validate(estimator=lrm,
                     X=X,
                     y=y,
                     cv=10,
                     scoring = ['accuracy', 'precision', 'r2'],
                     return_train_score=True
                    )

print('Test Kümesi Doğruluk Ortalaması     : {:.2f}'.format(cv['test_accuracy'].mean()))
print('Test Kümesi R-kare  Ortalaması      : {:.2f}'.format(cv['test_r2'].mean()))
print('Test Kümesi Hassasiyet Ortalaması   : {:.2f}'.format(cv['test_precision'].mean()))
print('Eğitim Kümesi Doğruluk Ortalaması   : {:.2f}'.format(cv['train_accuracy'].mean()))
print('Eğitim Kümesi R-kare  Ortalaması    : {:.2f}'.format(cv['train_r2'].mean()))
print('Eğitim Kümesi Hassasiyet Ortalaması : {:.2f}'.format(cv['train_precision'].mean()))

###########################################################################################################
# Grid Search

logreg = LogisticRegression()

parametreler = {"C": [10 ** x for x in range(-5, 5, 1)],
                "penalty": ['l1','l2']
                }


grid_cv = GridSearchCV(estimator=logreg,
                       param_grid = parametreler,
                       cv = 10,
                       # return_train_score=True
                      )
grid_cv.fit(X, y)
print("\nGrid Search değerleri:")
print("En iyi parametreler : ", grid_cv.best_params_)
print("En iyi skor         : ", grid_cv.best_score_)

sonuçlar = grid_cv.cv_results_
df = pd.DataFrame(sonuçlar)


df = df[['param_penalty','param_C', 'mean_test_score','params']]
df = df.sort_values(by='mean_test_score', ascending = False)
print(df)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(10,7))
sns.scatterplot(x = 'param_C', y = 'mean_test_score', hue = 'param_penalty', data = df[0:10], s=200)
plt.xscale('symlog')
plt.ylim((0.6,1))
plt.show()

###########################################################################################################
# Randomized Search CV

print("\nRandomized Search Değerleri")
parametreler = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2']
                }

rs_cv = RandomizedSearchCV(estimator=logreg,
                           param_distributions = parametreler,
                           cv = 10,
                           n_iter = 10,
                           random_state = 111,
                           scoring = 'precision'
                      )
rs_cv.fit(X, y)

print("En iyi parametreler        : ", rs_cv.best_params_)
print("Tüm hassasiyet değerleri   : ", rs_cv.cv_results_['mean_test_score'])
print("En iyi hassasiyet değeri   : ", rs_cv.best_score_)

sonuçlar_rs = rs_cv.cv_results_
df_rs = pd.DataFrame(sonuçlar_rs)

df_rs = df_rs[['param_penalty','param_C', 'mean_test_score','params']]
df_rs = df_rs.sort_values(by='mean_test_score', ascending = False)
print()
print(df_rs)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(9,6))
sns.scatterplot(x = 'param_C', y = 'mean_test_score', hue = 'param_penalty', data = df_rs, s=200)
plt.xscale('symlog')
plt.ylim((0.6,1))
plt.show()

# en iyi modelimiz grid search, C=0.1 ve l2'deyken bulundu.



