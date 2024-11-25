import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report
import numpy as np

# Carregar o arquivo CSV gerado
data = pd.read_csv('merged_jurassic_park_ratings.csv')

# Pré-processamento
data['rating_binary'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)  # Alta nota (1) se rating >= 4, caso contrário, Baixa nota (0)
features = data[['userId', 'movieId', 'timestamp']]  # Usaremos essas colunas como features

# Dividir o dataset para treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, data['rating_binary'], test_size=0.3, random_state=42)

# Para o modelo de regressão, usaremos a variável rating diretamente
y_train_reg = data.loc[X_train.index, 'rating']
y_test_reg = data.loc[X_test.index, 'rating']


# Modelo de Árvore de Decisão para Classificação Binária
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Avaliação
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_f1 = f1_score(y_test, y_pred_tree)

print("Árvore de Decisão - Acurácia:", tree_accuracy)
print("Árvore de Decisão - F1-Score:", tree_f1)
print(classification_report(y_test, y_pred_tree))


# Modelo de Rede Neural para Regressão
nn_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train_reg)
y_pred_nn = nn_model.predict(X_test)

# Avaliação (Métricas de Regressão)
nn_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_nn))
print("Rede Neural - RMSE:", nn_rmse)


# Modelo Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Avaliação
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)

print("Naive Bayes - Acurácia:", nb_accuracy)
print("Naive Bayes - F1-Score:", nb_f1)
print(classification_report(y_test, y_pred_nb))

print("Comparação dos Modelos:")
print("Árvore de Decisão - Acurácia:", tree_accuracy, "F1-Score:", tree_f1)
print("Rede Neural - RMSE:", nn_rmse)
print("Naive Bayes - Acurácia:", nb_accuracy, "F1-Score:", nb_f1)
