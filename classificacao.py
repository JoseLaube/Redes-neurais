import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv(
    "dados.txt",
    sep="\t",          # separador entre colunas
    decimal=",",       # vírgula decimal
    header=None,
    names=["X1", "X2", "label"]
)

duplicados = dados.duplicated()

print("Duplicados:", duplicados.sum())