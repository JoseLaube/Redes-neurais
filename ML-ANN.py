import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.optimizers import Adam


dados = pd.read_csv(
    "dados_15.txt",
    sep="\t",          # separador entre colunas
    decimal=",",       # vírgula decimal
    header=None,
    names=["X1", "X2", "label"]
)

print("\nEtapa 3: Tratamento de Dados Faltantes")

nulos = dados.isnull().sum()
print("Valores nulos por coluna:\n", nulos)

if nulos.sum() > 0:
    print("Removendo linhas com dados faltantes")
    dados = dados.dropna()
    # imputação de dados seria aqui se precisacemos:
    # dados.fillna(dados.median(), inplace=True)

X = dados[["X1", "X2"]].values
y = dados["label"].values

y = y - 1 # trocando por causa do índice 0 do tensorflow



print("\nEtapa 5: Separação dos Dados")

# Separação em 70% 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,  # 30% vai para as variaveis temps E 70% para as trains
    random_state=42, # semente
    stratify=y       # valores das classes
)

# Dos 30%, 15% teste e 15% validação
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,  # 30% * 0,5 = 15%
    random_state=42,
    stratify=y_temp
)

print(f"Tamanho do Treino: {len(X_train)} amostras")
print(f"Tamanho da Validação: {len(X_val)} amostras")
print(f"Tamanho do Teste: {len(X_test)} amostras")



print("\nEtapa 4: Escalonamento dos Dados")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Escala entre 0 e 1
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("DEBUG - ANTES do escalonamento:", X_train[0])
print("DEBUG - APÓS o escalonamento:", X_train_scaled[0])



print("\nEtapa 6: Construção da Rede Neural (MLP - Multi-Layer Perceptron)")

model = keras.Sequential()

model.add(layers.Dense(units=16, activation='relu', input_shape=(2,)))
# -> Camada Oculta
# - units=16: 16 neurônios na camada oculta
# - input_shape=(2,): indicamos que a entrada tem 2 atributos (X1 e X2)

model.add(layers.Dense(units=15, activation='softmax'))
# -> Camada de Saída
# - units=4: 4 neurônios, um para cada classe (Classe 1, 2, 3 e 4)
# - activation='softmax': converte a saída em um vetor de probabilidades (soma = 100%)

model.summary()



print("\n--- Etapa 7: Treinamento ---")

TAXA_APRENDIZADO = 0.003
otimizador_adam = Adam(learning_rate=TAXA_APRENDIZADO)

model.compile(
    optimizer=otimizador_adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# - loss='sparse_categorical_crossentropy': Função de perda para múltiplas classes com labels inteiros (0, 1, 2, 3)
# - metrics=['accuracy']: Mostra a acurácia (taxa de acerto) durante o treino


EPOCAS = 70
BATCH_SIZE = 16   # backpropagation a cada 16 amostras

history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val), # Avalia na validação ao fim de cada época
    epochs=EPOCAS,
    batch_size=BATCH_SIZE,
    verbose=1 # Mostra a barra de progresso no terminal
)

print("\nTreinamento concluído")



print("\nEtapa 8: Gerando Curvas de Aprendizado")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
dados_historico = history.history

# Gráfico da acuracia (Taxa de Acerto)
ax[0].plot(dados_historico['accuracy'], label='Treino', color='blue', linewidth=2)
ax[0].plot(dados_historico['val_accuracy'], label='Validação', color='orange', linestyle='--', linewidth=2)
ax[0].set_title('Evolução da Acurácia (Convergência)', fontsize=14)
ax[0].set_xlabel('Épocas', fontsize=12)
ax[0].set_ylabel('Acurácia', fontsize=12)
ax[0].legend(loc='lower right')
ax[0].grid(True, linestyle=':', alpha=0.7)

# Gráfico da perda
ax[1].plot(dados_historico['loss'], label='Treino', color='blue', linewidth=2)
ax[1].plot(dados_historico['val_loss'], label='Validação', color='orange', linestyle='--', linewidth=2)
ax[1].set_title('Evolução da Perda (Loss)', fontsize=14)
ax[1].set_xlabel('Épocas', fontsize=12)
ax[1].set_ylabel('Perda', fontsize=12)
ax[1].legend(loc='upper right')
ax[1].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig("imagens/curvas_aprendizado.png", dpi=300)



print("\nEtapa 9: Avaliação Final")

# .predict() solta as probabilidades (ex: [0.01, 0.98, 0.005, 0.005])
y_pred_probabilidades = model.predict(X_test_scaled)

# O np.argmax pega a posição da maior probabilidade (no ex acima, posição 1)
y_pred = np.argmax(y_pred_probabilidades, axis=1)


# RELATÓRIO:
# nomes_classes = ["Classe 1", "Classe 2", "Classe 3", "Classe 4"]
nomes_classes = ["Classe 1", "Classe 2", "Classe 3", "Classe 4","Classe 5", "Classe 6","Classe 7", "Classe 8", "Classe 9", "Classe 10", "Classe 11", "Classe 12", "Classe 13", "Classe 14", "Classe 15"]
print("\nRelatório de Classificação:")
relatorio = classification_report(y_test, y_pred, target_names=nomes_classes)
print(relatorio)

# Compara o gabarito (y_test) com as respostas do modelo (y_pred)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)

disp.plot(cmap='Blues', ax=ax, values_format='d')

ax.set_title('Matriz de Confusão - Dados de Teste', fontsize=14)
ax.set_xlabel('Previsão do Modelo (Predicted Class)', fontsize=12)
ax.set_ylabel('Classe Real (Actual Class)', fontsize=12)

plt.tight_layout()
plt.savefig("imagens/matriz_confusao.png", dpi=300)

print("Código finalizado com sucesso")
