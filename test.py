import matplotlib.pyplot as plt

# Listas para armazenar os dados de x e y
x = []
y = []

# Ler o arquivo de dados
with open('data.txt', 'r') as file:
    lines = file.readlines()[1110:5871]  # Selecionar as linhas desejadas

    for line in lines:
        values = line.strip().split()
        if len(values) >= 2:
            x.append(float(values[0]))
            y.append(float(values[1]))

# Criar o gráfico de dispersão (scatter plot)
plt.scatter(x, y, marker='o', s=10)  # Customize o estilo do gráfico conforme necessário
# plt.hexbin(x, y, gridsize=50, cmap='Greys', extent=(-0.005, 0.005, -0.0025, 0.0025))  # Ajuste os parâmetros conforme necessário
plt.xlabel('Posição X')
plt.ylabel('Posição Y')
plt.title('Gráfico de Dispersão de Posição com período de 3 s')

# Adicione uma barra de cores
# plt.colorbar(label='Contagem')

# Mostrar o gráfico
plt.show()