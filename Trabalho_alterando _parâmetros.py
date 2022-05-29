# Alunos: Armando Erick,Iago Oliveira Guedes, Warley Coutinho.


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

dados = pd.read_csv("./Vestments.csv")

data['Vestments','name']

data.head()

dados.insert('Vestments',['name'].str.slice(15,  91).str.lower())

dados['EstimatedSalary'] = dados['EstimatedSalary'].str.title()


dados['Purchased'] = dados['Purchased'].str.replace(" ", "")

dados