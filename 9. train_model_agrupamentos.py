import calendar
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import os
import gc

# Função que retorna o número exato de dias em um mês específico.
# Utiliza a biblioteca 'calendar' para calcular o número de dias com base no mês e ano fornecidos.
def get_days_in_month(month, year):
    return calendar.monthrange(year, month)[1]

# Função que cria variáveis temporais cíclicas no DataFrame.
# Extrai informações de ano, mês, dia, hora e minuto de uma coluna com datas no formato ISO 8601.
# Adiciona variáveis senoidais e cossenoidais para representar características cíclicas de tempo.
def create_time_features(df):
    # Extração das partes de tempo a partir do formato ISO 8601
    df['year'] = pd.to_datetime(df['DataHora_ISO8601']).dt.year
    df['mes'] = pd.to_datetime(df['DataHora_ISO8601']).dt.month
    df['dia'] = pd.to_datetime(df['DataHora_ISO8601']).dt.day
    df['hora'] = pd.to_datetime(df['DataHora_ISO8601']).dt.hour
    df['minuto'] = pd.to_datetime(df['DataHora_ISO8601']).dt.minute

    # Cálculo das variáveis cíclicas usando funções trigonométricas
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / df.apply(lambda row: get_days_in_month(row['mes'], row['year']), axis=1))
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / df.apply(lambda row: get_days_in_month(row['mes'], row['year']), axis=1))
    df['dia_sin'] = np.sin(2 * np.pi * df['dia'] / 31)
    df['dia_cos'] = np.cos(2 * np.pi * df['dia'] / 31)
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['minuto_sin'] = np.sin(2 * np.pi * df['minuto'] / 60)
    df['minuto_cos'] = np.cos(2 * np.pi * df['minuto'] / 60)

    return df

# Função que cria uma estrutura de árvore KD (KDTree) para encontrar grupos próximos com base em coordenadas.
# Constrói a KDTree a partir de uma lista de dicionários com latitudes e longitudes.
def build_kdtree(groups):
    coords = [(group['lat'], group['long']) for group in groups]
    return KDTree(np.array(coords))

# Função para localizar o grupo mais próximo utilizando a estrutura KDTree.
# Retorna o índice do grupo mais próximo se a distância estiver dentro de uma tolerância especificada.
def find_nearest_group(kdtree, lat, long, tolerance=0.002):
    distance, index = kdtree.query([lat, long])
    return index if distance <= tolerance else None

# Função que processa grandes arquivos CSV em batches (lotes) e salva agrupamentos em arquivos CSV separados.
# Realiza o pré-processamento, cria características temporais, e organiza os dados em agrupamentos geográficos.
def process_and_save_batches(csv_file, batch_size=5000000):
    # Definição do diretório de saída para os arquivos agrupados
    output_dir = os.path.dirname(csv_file)
    group_dir = os.path.join(output_dir, 'Agrupamentos')
    os.makedirs(group_dir, exist_ok=True)

    group_mapping = []  # Lista para armazenar grupos criados

    # Leitura dos dados em batches para evitar sobrecarga de memória
    for batch_idx, chunk in enumerate(pd.read_csv(csv_file, chunksize=batch_size)):
        print(f"Lendo batch {batch_idx + 1}")

        # Conversão da coluna de data/hora para o formato datetime
        chunk['DataHora_ISO8601'] = pd.to_datetime(chunk['DataHora_ISO8601'], errors='coerce')

        # Aplicação da função de criação de variáveis temporais cíclicas
        chunk = create_time_features(chunk)

        # Construção da KDTree com os grupos existentes (se houver)
        kdtree = build_kdtree(group_mapping) if group_mapping else None

        # Iteração sobre os dados do batch
        for _, row in chunk.iterrows():
            lat, long = row['lat'], row['long']

            # Verificação do grupo mais próximo ou criação de um novo grupo
            group_idx = find_nearest_group(kdtree, lat, long) if kdtree else None

            if group_idx is not None:
                group_mapping[group_idx]['data'].append(row)
            else:
                new_group = {'lat': lat, 'long': long, 'data': [row]}
                group_mapping.append(new_group)

    # Salvamento dos grupos em arquivos CSV separados
    for idx, group in enumerate(group_mapping):
        group_df = pd.DataFrame(group['data'])
        group_csv_path = os.path.join(group_dir, f'agrupamento_{idx + 1}.csv')

        # Adiciona ou cria o arquivo CSV correspondente
        if not os.path.exists(group_csv_path):
            group_df.to_csv(group_csv_path, index=False)
        else:
            group_df.to_csv(group_csv_path, mode='a', header=False, index=False)

        # Limpeza de memória
        del chunk
        gc.collect()

    print("Processamento completo e CSVs dos grupos foram salvos.")
    return group_mapping

# Função para criar uma interface gráfica simples para seleção de arquivos.
# Utiliza a biblioteca Tkinter para criar a interface e executar o processamento de agrupamentos.
def create_gui():
    from tkinter import filedialog, messagebox
    import tkinter as tk

    # Função para selecionar o arquivo CSV
    def select_file():
        csv_file = filedialog.askopenfilename(filetypes=[("Arquivos CSV", "*.csv")])
        if csv_file:
            try:
                process_and_save_batches(csv_file)
                messagebox.showinfo("Sucesso", "Agrupamento concluído.")
            except Exception as e:
                messagebox.showerror("Erro", f"Ocorreu um erro: {e}")

    # Criação da janela principal da interface
    root = tk.Tk()
    root.title("Processamento de Agrupamentos")
    tk.Label(root, text="Selecione o arquivo CSV:").pack(padx=10, pady=5)
    tk.Button(root, text="Procurar", command=select_file).pack(padx=10, pady=10)
    root.mainloop()

# Inicia a execução da interface gráfica quando o script é executado diretamente.
if __name__ == "__main__":
    create_gui()
