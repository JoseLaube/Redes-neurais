import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("imagens", exist_ok=True)

dados = pd.read_csv(
    "dados_15.txt",
    sep="\t",          # separador entre colunas
    decimal=",",       # vírgula decimal
    header=None,
    names=["X1", "X2", "label"]
)



# Gráfico de dispersão
plt.figure(figsize=(12, 8))

classes_unicas = sorted(dados["label"].unique())

cmap = plt.get_cmap('tab20')
cores_dinamicas = [cmap(i) for i in np.linspace(0, 1, len(classes_unicas))]

for i, classe in enumerate(classes_unicas):

    subset = dados[dados["label"] == classe]

    plt.scatter(
        subset["X1"],
        subset["X2"],
        label=f"Classe {classe}",
        color=cores_dinamicas[i],
        alpha=0.7
    )

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Distribuição dos dados")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.savefig("imagens/distribuicao_dados.png", dpi=300)
plt.show()



# Histogramas
fig, ax = plt.subplots(1, 2, figsize=(12,8))

for classe in sorted(dados["label"].unique()):

    subset = dados[dados["label"] == classe]

    ax[0].hist(
        subset["X1"],
        bins=15,
        alpha=0.5,
        label=f"Classe {classe}"
    )

    ax[1].hist(
        subset["X2"],
        bins=15,
        alpha=0.5,
        label=f"Classe {classe}"
    )

ax[0].set_title("X1 por classe")
ax[1].set_title("X2 por classe")
ax[0].legend()
ax[1].legend()

plt.tight_layout()
plt.savefig("imagens/histogramas_dados.png", dpi=300)
plt.show()



# Boxplots
fig, ax = plt.subplots(1, 2, figsize=(12,6))

dados.boxplot(
    column="X1",
    by="label",
    ax=ax[0]
)

dados.boxplot(
    column="X2",
    by="label",
    ax=ax[1]
)

ax[0].set_title("X1")
ax[1].set_title("X2")

plt.suptitle("Distribuição dos Atributos por Classe")
plt.tight_layout()
plt.savefig("imagens/boxplots_dados.png", dpi=300)
plt.show()



# Verificação de dados duplicados
duplicados = dados.duplicated()
print("Duplicados:", duplicados.sum())