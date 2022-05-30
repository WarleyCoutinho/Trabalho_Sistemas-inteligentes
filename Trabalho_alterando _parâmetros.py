# Alunos: Armando Erick, Iago Oliveira Guedes, Warley Coutinho.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


dados = pd.read_csv("./Vestments.csv")

x = dados[['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'diabetes', 'age']]
y = dados['test']

xtreino, xteste, ytreino, yteste = train_test_split(x, y, random_state=1)

svm = SVC()
svm = svm.fit(xtreino, ytreino)
yprevisao = svm.predict(xteste)

svm.support_vectors_
svm.n_support_

metrics.accuracy_score(yteste, yprevisao)

X = dados.values[:100, 1:3]
y = dados.values[:100, 8].astype(np.integer)

svm = SVC(kernel='rbf')
svm.fit(X, y)

plot_decision_regions(X, y, clf=svm)

svm = LinearSVC()
svm.fit(X, y)

plot_decision_regions(X, y, clf=svm)
plt.show()

#Scaling


dados = pd.read_csv("./Vestments.csv")

x = dados[['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'diabetes', 'age']]
y = dados['test']

xtreino, xteste, ytreino, yteste = train_test_split(x, y, random_state=1)

xtreino.min()

xtreino_range = xtreino.max() - xtreino.min()

xtreino_padroniz = (xtreino - xtreino.min()) / xtreino_range

ytreino_range = ytreino.max() - ytreino.min()
ytreino_padroniz = (ytreino - ytreino.min()) / ytreino_range

xteste_range = xteste.max() - xteste.min()
xteste_padroniz = (xteste - xteste.min()) / xteste_range

X = xtreino_padroniz.values[:, 1:3]
y = ytreino.values.astype(np.integer)

svm = SVC(kernel='linear')
svm.fit(X, y)

plot_decision_regions(X, y, clf=svm, legend=2)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('SVM com Kernel Linear - Tipos de Flores')

svm = SVC()
svm = svm.fit(xtreino_padroniz, ytreino)

yprevisao = svm.predict(xteste_padroniz)
yprevisao[0:5]

cm = np.array(confusion_matrix(yteste,yprevisao, labels=[1,0]))
avaliacao = pd.DataFrame(cm,index=['tem diabetes', 'está saudável'], columns=['previsto com diabetes', 'previsto como saudável'])

metrics.accuracy_score(yteste, yprevisao)