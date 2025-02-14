import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib

# Configuração do backend do Matplotlib para evitar dependências de Tkinter em servidores
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import json

# Definição do diretório fixo onde os arquivos de agrupamentos estão localizados
GROUP_DIR = r"D:\Projetos\Resultados\Treino_Agrupamentos"

# Função que cria variáveis temporais cíclicas no DataFrame
# Representa características temporais como mês, dia, hora e minuto usando seno e cosseno
# para capturar periodicidade em modelos de aprendizado

def create_time_features(df):
    """
    Adiciona variáveis temporais cíclicas ao DataFrame para capturar periodicidade temporal.
    
    Parâmetros:
    - df (DataFrame): Dados de entrada contendo colunas de tempo.

    Retorna:
    - DataFrame atualizado com colunas de características cíclicas.
    """
    df['year'] = pd.to_datetime(df['DataHora_ISO8601']).dt.year
    df['mes'] = pd.to_datetime(df['DataHora_ISO8601']).dt.month
    df['dia'] = pd.to_datetime(df['DataHora_ISO8601']).dt.day
    df['hora'] = pd.to_datetime(df['DataHora_ISO8601']).dt.hour
    df['minuto'] = pd.to_datetime(df['DataHora_ISO8601']).dt.minute

    # Geração de características cíclicas
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_sin'] = np.sin(2 * np.pi * df['dia'] / 31)
    df['dia_cos'] = np.cos(2 * np.pi * df['dia'] / 31)
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['minuto_sin'] = np.sin(2 * np.pi * df['minuto'] / 60)
    df['minuto_cos'] = np.cos(2 * np.pi * df['minuto'] / 60)

    return df

# Função para treinar um modelo de regressão linear em um agrupamento específico
# Normaliza as variáveis preditoras e calcula métricas de desempenho do modelo

def train_linear_model(group_csv_path, output_dir, summary_path):
    """
    Treina um modelo de regressão linear para o agrupamento especificado e salva os resultados.
    
    Parâmetros:
    - group_csv_path (str): Caminho para o arquivo CSV do agrupamento.
    - output_dir (str): Diretório onde os resultados serão salvos.
    - summary_path (str): Caminho para o arquivo de resumo dos agrupamentos.
    """
    # Leitura e pré-processamento dos dados
    group_df = pd.read_csv(group_csv_path)
    group_df = create_time_features(group_df)

    # Seleção de características e variável alvo
    X = group_df[['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos',
                  'hora_sin', 'hora_cos', 'minuto_sin', 'minuto_cos']]
    y = group_df['Volume de Carros']

    # Normalização das variáveis preditoras
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Criação e treinamento do modelo
    model = LinearRegression()
    model.fit(X, y)

    # Previsões e cálculo do erro médio absoluto (MAE)
    predictions = model.predict(X)
    mae = np.mean(np.abs(predictions - y))

    # Salvar o modelo em formato JSON
    model_name = os.path.basename(group_csv_path).replace('.csv', '_linear_model.json')
    with open(os.path.join(output_dir, model_name), 'w') as f:
        json.dump(model.get_params(), f)

    # Salvar estatísticas e gráficos
    mean_volume = group_df['Volume de Carros'].mean()
    statistics = {"mean_volume": mean_volume, "mae": mae}
    save_training_history_and_plots(predictions, y, statistics, output_dir)

    # Atualizar o resumo do agrupamento
    update_group_summary_txt(summary_path, group_csv_path, group_df)

# Função para atualizar o resumo dos agrupamentos com coordenadas mínimas e máximas

def update_group_summary_txt(summary_path, group_csv_path, group_df):
    """
    Atualiza o arquivo de resumo com os limites de coordenadas de um agrupamento.
    
    Parâmetros:
    - summary_path (str): Caminho para o arquivo de resumo.
    - group_csv_path (str): Caminho para o arquivo CSV do agrupamento.
    - group_df (DataFrame): Dados do agrupamento.
    """
    min_lat, max_lat = group_df['lat'].min(), group_df['lat'].max()
    min_long, max_long = group_df['long'].min(), group_df['long'].max()

    with open(summary_path, 'a') as f:
        f.write(f"{os.path.basename(group_csv_path)}: "
                f"LAT [{min_lat}, {max_lat}], LONG [{min_long}, {max_long}]\n")

# Função para salvar as estatísticas e gráficos de comparação entre previsões e valores reais

def save_training_history_and_plots(predictions, y, statistics, folder_path):
    """
    Salva gráficos e estatísticas do modelo em arquivos específicos.
    
    Parâmetros:
    - predictions (array): Valores previstos pelo modelo.
    - y (array): Valores reais observados.
    - statistics (dict): Estatísticas calculadas.
    - folder_path (str): Caminho para salvar os arquivos gerados.
    """
    # Salvar estatísticas em um arquivo JSON
    with open(os.path.join(folder_path, 'statistics.json'), 'w') as f:
        json.dump(statistics, f)

    # Gerar e salvar gráfico comparativo
    plt.figure()
    plt.plot(y, label='Real', marker='o')
    plt.plot(predictions, label='Previsão', marker='x')
    plt.title('Comparação entre Previsão e Valores Reais')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'prediction_vs_real.png'))
    plt.close()

# Função principal que treina modelos para todos os agrupamentos no diretório definido

def train_all_groups():
    """
    Percorre todos os arquivos CSV no diretório definido e treina modelos para cada agrupamento.
    """
    summary_path = os.path.join(GROUP_DIR, 'agrupamentos_summary.txt')

    for group_csv in os.listdir(GROUP_DIR):
        if group_csv.endswith('.csv'):
            print(f'Treinando agrupamento: {group_csv}')
            output_dir = os.path.join(GROUP_DIR, group_csv.replace('.csv', ''))
            os.makedirs(output_dir, exist_ok=True)
            train_linear_model(os.path.join(GROUP_DIR, group_csv), output_dir, summary_path)

# Execução do treinamento se o script for executado diretamente
if __name__ == "__main__":
    train_all_groups()
    print("Treinamento concluído para todos os grupos.")
