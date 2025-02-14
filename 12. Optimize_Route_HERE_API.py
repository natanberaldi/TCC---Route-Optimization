import os
import random
import tkinter as tk
from tkinter import messagebox
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib
import joblib
from flexpolyline import decode as decode_flexible_polyline

# =============================================================================
# CONFIGURAÇÕES GERAIS (CHAVES DE API E DIRETÓRIOS)
# =============================================================================

# Chave de acesso à API HERE.
HERE_API_KEY = "pFkT_7Exa59sbAQxMcgTP06x1tdUom8SQjQBfyPti3g"
# Chave de acesso à API Google.
GOOGLE_API_KEY = "AIzaSyCAaAqMbRo1B4aNUDWlT-qTQcl6j2nfDIw"
# Diretório onde estão armazenados os modelos de cada agrupamento.
MODEL_DIR = r"D:\Projetos\Resultados\Treino_Agrupamentos"
# Caminho do arquivo que resume limites geográficos (bounding boxes) de cada agrupamento.
SUMMARY_FILE = os.path.join(MODEL_DIR, 'agrupamentos_summary.txt')

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def convert_to_cyclic_features(dia, mes, hora, minuto):
    """
    Função que cria variáveis cíclicas (seno e cosseno) a partir de valores de dia, mês, hora e minuto.
    Utiliza a biblioteca 'numpy' para produzir a representação trigonométrica que auxilia no reconhecimento
    de padrões temporais.
    """
    data = {
        'mes_sin': np.sin(2 * np.pi * mes / 12),
        'mes_cos': np.cos(2 * np.pi * mes / 12),
        'dia_sin': np.sin(2 * np.pi * dia / 31),
        'dia_cos': np.cos(2 * np.pi * dia / 31),
        'hora_sin': np.sin(2 * np.pi * hora / 24),
        'hora_cos': np.cos(2 * np.pi * hora / 24),
        'minuto_sin': np.sin(2 * np.pi * minuto / 60),
        'minuto_cos': np.cos(2 * np.pi * minuto / 60)
    }
    return pd.DataFrame([data])

def reverse_geocode_here(lat, lon):
    """
    Função que realiza geocodificação reversa utilizando a API HERE.
    Retorna o endereço mais próximo com base em latitude e longitude.
    """
    try:
        url = "https://revgeocode.search.hereapi.com/v1/revgeocode"
        params = {
            "at": f"{lat},{lon}",
            "lang": "pt-BR",
            "apikey": HERE_API_KEY
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "items" in data and len(data["items"]) > 0:
            return data["items"][0]["address"].get("label", "")
        else:
            print(f"[reverse_geocode_here] Nenhum resultado para ({lat},{lon})")
    except requests.RequestException as e:
        print(f"[reverse_geocode_here] Erro: {e}")
    return None

def geocode_google(address):
    """
    Função que realiza geocodificação direta usando a API do Google.
    Dado um endereço textual, retorna a latitude e longitude aproximadas.
    """
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
    except requests.RequestException as e:
        print(f"[geocode_google] Erro: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[geocode_google] Erro ao processar resposta: {e}")
    return None

def find_group_for_coordinates(lat, lon, delta=0.01):
    """
    Função que identifica o agrupamento correspondente a determinada coordenada (lat, lon),
    comparando-a com os valores mínimos e máximos de cada bounding box em 'agrupamentos_summary.txt'.
    Se a coordenada estiver dentro do delta especificado, retorna o nome do agrupamento.
    """
    A_min_lat = lat - delta
    A_max_lat = lat + delta
    A_min_lon = lon - delta
    A_max_lon = lon + delta

    try:
        with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    group_name, ranges = line.split(': ')
                    group_name = group_name.replace(".csv", "").strip()

                    lat_part, lon_part = ranges.split(', LONG')
                    lat_part = lat_part.replace("LAT [", "").replace("]", "").strip()
                    lon_part = lon_part.replace("[", "").replace("]", "").strip()

                    min_lat, max_lat = map(float, lat_part.split(','))
                    min_lon, max_lon = map(float, lon_part.split(','))

                    # bounding box do summary
                    B_min_lat, B_max_lat = min_lat, max_lat
                    B_min_lon, B_max_lon = min_lon, max_lon

                    lat_overlap = (A_min_lat <= B_max_lat) and (B_min_lat <= A_max_lat)
                    lon_overlap = (A_min_lon <= B_max_lon) and (B_min_lon <= A_max_lon)

                    if lat_overlap and lon_overlap:
                        return group_name
                except ValueError as e:
                    print(f"[find_group_for_coordinates] Linha mal formatada: {line} - {e}")
    except FileNotFoundError:
        print(f"[find_group_for_coordinates] Arquivo {SUMMARY_FILE} não encontrado.")

    return None

def load_group_model(group_name):
    """
    Função que carrega o modelo de Regressão Linear e o objeto de normalização (scaler),
    referentes a um agrupamento específico, utilizando a biblioteca 'joblib'.
    """
    group_name = group_name.replace(".csv", "")
    model_path = os.path.join(MODEL_DIR, group_name, f"{group_name}_linear_model.pkl")

    if not os.path.exists(model_path):
        print(f"[load_group_model] Modelo não encontrado: {model_path}")
        return None

    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"[load_group_model] Erro ao carregar o modelo {model_path}: {e}")
        return None

def load_group_statistics(group_name):
    """
    Função que carrega as estatísticas de desempenho e o gráfico de previsão gerados
    pelo treinamento de um agrupamento (caso existam). Retorna um dicionário de estatísticas
    e o nome do arquivo de gráfico (.png) ou None, se não encontrado.
    """
    group_dir = os.path.join(MODEL_DIR, group_name)
    stats_file = os.path.join(group_dir, "statistics.json")
    statistics = {}
    graph_file = None

    # Leitura do arquivo de estatísticas
    try:
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                statistics = json.load(f)
        else:
            print(f"[load_group_statistics] statistics.json não encontrado em {group_dir}")
    except Exception as e:
        print(f"[load_group_statistics] Erro ao carregar statistics.json: {e}")
        statistics = {}

    # Localização de um arquivo .png relacionado ao agrupamento
    try:
        if os.path.exists(group_dir):
            for file in os.listdir(group_dir):
                if file.endswith(".png"):
                    graph_file = file
                    break
    except Exception as e:
        print(f"[load_group_statistics] Erro ao buscar .png: {e}")
        graph_file = None

    return statistics, graph_file

# =============================================================================
# PROCESSAMENTO DE COORDENADAS
# =============================================================================

def process_single_coordinate(lat, lon, dia, mes, hora, minuto):
    """
    Função que executa geocodificação reversa (HERE), geocodificação direta (Google),
    identifica a qual agrupamento a coordenada pertence, carrega o modelo linear correspondente
    e prevê o volume de tráfego. Retorna uma tupla com (lat, lon, volume_previsto, endereço,
    nome_do_grupo, estatísticas_do_grupo, nome_do_arquivo_de_gráfico).
    """
    try:
        address = reverse_geocode_here(lat, lon)
        if not address:
            return lat, lon, 0, None, None, {}, None

        google_latlon = geocode_google(address)
        if not google_latlon:
            return lat, lon, 0, address, None, {}, None
        google_lat, google_lon = google_latlon

        group_name = find_group_for_coordinates(google_lat, google_lon)
        if not group_name:
            return lat, lon, 0, address, None, {}, None

        model_data = load_group_model(group_name)
        if not model_data:
            return lat, lon, 0, address, group_name, {}, None

        # Criação das variáveis temporais cíclicas
        features_df = convert_to_cyclic_features(dia, mes, hora, minuto)

        # Normalização dos dados e predição
        features_scaled = model_data["scaler"].transform(features_df)
        traffic_volume = model_data["model"].predict(features_scaled)[0]

        statistics, graph_file = load_group_statistics(group_name)

        return lat, lon, traffic_volume, address, group_name, statistics, graph_file

    except Exception as e:
        print(f"[process_single_coordinate] Erro: {e}")
        return lat, lon, 0, None, None, {}, None

def calculate_mae(y_real, y_pred):
    """
    Função que calcula o MAE (Mean Absolute Error) entre valores reais e previstos,
    retornando 'N/A' se não houver dados suficientes.
    """
    if len(y_real) == 0 or len(y_pred) == 0:
        return "N/A"
    return np.mean(np.abs(np.array(y_real) - np.array(y_pred)))

def process_coordinates(route_coordinates, dia, mes, hora, minuto, avoid_coords):
    """
    Função que processa uma lista de coordenadas (rota) em paralelo, utilizando
    ThreadPoolExecutor. Para cada coordenada, chama 'process_single_coordinate'
    e agrega as previsões de volume de tráfego, aplicando penalidade a trechos acima
    de um certo valor. Retorna os resultados processados e estatísticas sobre o erro.
    """
    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            total_coordinates = len(route_coordinates)
            coordinates_found_in_summary = 0
            penalties_applied = 0

            group_volumes = {}

            # Submete lotes de coordenadas (10 em 10) para processamento simultâneo
            for i in range(0, total_coordinates, 10):
                batch = route_coordinates[i:i+10]
                for (lat, lon) in batch:
                    futures.append(executor.submit(
                        process_single_coordinate, lat, lon, dia, mes, hora, minuto
                    ))

            coordenadas_processadas = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        (lat, lon, traffic_volume, address,
                         group_name, statistics, graph_file) = result
                        coordenadas_processadas.append(result)

                        # Se houver volume previsto > 0, significa que foi encontrado no summary
                        if traffic_volume > 0:
                            coordinates_found_in_summary += 1

                        # Se o volume ultrapassar 80, insere penalidade (bounding box para evitar rota)
                        if traffic_volume > 80:
                            penalties_applied += 1
                            d = 0.01
                            polygon = [
                                [lon - d, lat - d],
                                [lon + d, lat - d],
                                [lon + d, lat + d],
                                [lon - d, lat + d],
                                [lon - d, lat - d]
                            ]
                            avoid_coords.append(polygon)

                        # Cálculo de erro absoluto entre previsão e volume médio do agrupamento
                        mean_volume = statistics.get("mean_volume", 0)
                        if group_name not in group_volumes:
                            group_volumes[group_name] = {"abs_errors": []}

                        if mean_volume > 0:
                            abs_error = abs(traffic_volume - mean_volume)
                            group_volumes[group_name]["abs_errors"].append(abs_error)

                except Exception as e:
                    print(f"[process_coordinates] Erro ao processar coord: {e}")

            # Conclusão: cálculo do MAE por agrupamento
            group_maes = {
                group: np.mean(data["abs_errors"]) if data["abs_errors"] else "N/A"
                for group, data in group_volumes.items()
            }

            return (
                coordenadas_processadas,
                total_coordinates,
                coordinates_found_in_summary,
                penalties_applied,
                group_maes
            )

    except Exception as e:
        print(f"[process_coordinates] Erro: {e}")
        return [], 0, 0, 0, {}

# =============================================================================
# CHAMADA À API HERE (Routing v8)
# =============================================================================

def get_coordinates(address):
    """
    Função que faz a geocodificação (lat, lon) de um endereço usando a API HERE,
    retornando tupla (latitude, longitude) ou None se falhar.
    """
    try:
        url = "https://geocode.search.hereapi.com/v1/geocode"
        params = {
            "q": address,
            "apiKey": HERE_API_KEY
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            position = data["items"][0]["position"]
            return position["lat"], position["lng"]
        else:
            print(f"[get_coordinates] Nenhum resultado para: {address}")
    except requests.RequestException as e:
        print(f"[get_coordinates] Erro: {e}")
    return None

def decode_polyline(polyline_str):
    """
    Função que decodifica 'flexible polyline' (padrão da HERE Routing v8)
    usando a biblioteca 'flexpolyline'.
    Retorna lista de coordenadas no formato (lat, lon) ou (lat, lon, ele).
    """
    coords = decode_flexible_polyline(polyline_str)

    print("=== Lista de coordenadas decodificadas (Flexible) ===")
    for c in coords:
        print(c)
    print("==========================================\n")

    return coords

def consolidate_route_data(response_data):
    """
    Função que consolida a duração total, distância total e instruções de todas as seções
    em um dicionário retornado pela API HERE. Útil para extrair informações resumidas da rota.
    """
    total_duration = 0
    total_distance = 0
    all_instructions = []

    for route in response_data.get("routes", []):
        for section in route.get("sections", []):
            total_duration += section["summary"]["duration"]
            total_distance += section["summary"]["length"]
            all_instructions.extend([
                action["instruction"] for action in section.get("actions", [])
            ])

    return total_duration, total_distance, all_instructions

def calculate_route_with_waypoint(origin, destination, waypoint=None, avoid_coords=[]):
    """
    Função que utiliza a API HERE v8 para calcular a rota entre origem e destino,
    opcionalmente incluindo um waypoint. Também permite evitar regiões adicionadas
    em 'avoid_coords'. Retorna duração total, distância total, lista de instruções,
    parâmetros de requisição e dados completos da resposta.
    """
    origin_coords = get_coordinates(origin)
    destination_coords = get_coordinates(destination)
    if origin_coords is None or destination_coords is None:
        print("[calculate_route_with_waypoint] Origem ou destino não encontrados.")
        return None, None, [], {}, {}

    points = []
    points.append(f"{origin_coords[0]},{origin_coords[1]}")  # origem
    if waypoint:
        points.append(f"{waypoint[0]},{waypoint[1]}")        # waypoint
    points.append(f"{destination_coords[0]},{destination_coords[1]}")  # destino

    url = "https://router.hereapi.com/v8/routes"
    params = {
        "apiKey": HERE_API_KEY,
        "transportMode": "car",
        "return": "summary,polyline,actions,instructions",
        "lang": "pt-BR"
    }
    params["origin"] = points[0]
    params["destination"] = points[-1]
    if len(points) == 3:  # Se há um waypoint intermediário
        params["via"] = points[1]

    # Criação do payload para evitar áreas
    payload = {}
    if avoid_coords:
        avoid_boxes = []
        for poly in avoid_coords:
            lons = [p[0] for p in poly]
            lats = [p[1] for p in poly]
            avoid_boxes.append({
                "type": "boundingBox",
                "west": min(lons),
                "east": max(lons),
                "south": min(lats),
                "north": max(lats),
            })
        payload["avoid"] = {"areas": avoid_boxes}

    try:
        response = requests.post(url, params=params, json=payload)
        response.raise_for_status()
        data = response.json()

        if "routes" not in data or not data["routes"]:
            print("[calculate_route_with_waypoint] Nenhuma rota encontrada.")
            return None, None, [], params, data

        total_duration = 0
        total_distance = 0
        all_instructions = []

        for route in data["routes"]:
            for section in route.get("sections", []):
                total_duration += section["summary"]["duration"]
                total_distance += section["summary"]["length"]
                for action in section.get("actions", []):
                    all_instructions.append(action.get("instruction", "Sem instrução"))

        return total_duration / 60, total_distance, all_instructions, params, data

    except requests.RequestException as e:
        print(f"[calculate_route_with_waypoint] Erro na requisição: {e}")
        return None, None, [], payload, {}

# =============================================================================
# LOG E EXPORTAÇÃO DE RESULTADOS
# =============================================================================

def generate_random_filename():
    """
    Função que gera um nome de arquivo aleatório (para salvar relatórios de rotas)
    com base em um número randômico.
    """
    return f"resultado_rota_{random.randint(1000, 9999)}.txt"

def export_all_variables_to_txt(filename, **variables):
    """
    Função que exporta variáveis e seções descritivas de processamento em um arquivo TXT,
    para registro e auditoria de informações. Agrupa texto de forma organizada para cada seção.
    """
    try:
        with open(filename, "a", encoding="utf-8") as file:
            file.write("========= INÍCIO DO PROCESSAMENTO =========\n")
            file.write(f"Data/Hora de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for section, content in variables.items():
                file.write(f"### {section.upper()} ###\n")
                if isinstance(content, list):
                    for item in content:
                        file.write(f"- {item}\n")
                elif isinstance(content, dict):
                    for key, value in content.items():
                        file.write(f"{key}: {value}\n")
                else:
                    file.write(f"{content}\n")
                file.write("\n")
            file.write("========= FIM DO PROCESSAMENTO =========\n")
    except Exception as e:
        print(f"[export_all_variables_to_txt] Erro: {e}")

# =============================================================================
# FUNÇÃO PRINCIPAL (OTIMIZAÇÃO DE ROTA)
# =============================================================================

def optimize_route_with_waypoint(origin, destination, date, hour, waypoint=None):
    """
    Função principal que orquestra:
    1) Cálculo inicial de rota (HERE API).
    2) Processamento das coordenadas em busca de previsões de tráfego.
    3) Recalcula rota final incluindo penalidades (áreas a serem evitadas).
    4) Exporta todos os resultados para um arquivo TXT de relatório.
    """
    avoid_coords = []
    datahora = datetime.strptime(f"{date} {hour}", "%Y-%m-%d %H:%M")
    dia, mes, hora_, minuto = datahora.day, datahora.month, datahora.hour, datahora.minute

    # Rota inicial
    (initial_duration, initial_distance, rota_inicial_instructions,
     payload_inicial, response_inicial) = calculate_route_with_waypoint(
         origin, destination, waypoint, avoid_coords
    )
    if initial_duration is None:
        print("[optimize_route_with_waypoint] Erro: falha ao calcular rota inicial.")
        return

    print(f"Duração inicial: {initial_duration} minutos, Distância inicial: {initial_distance} metros")
    for instr in rota_inicial_instructions:
        print(" -", instr)

    # Processar coordenadas da rota inicial
    coordenadas_processadas = []
    total_coordinates = 0
    coordinates_found_in_summary = 0
    penalties_applied = 0
    group_maes = {}

    if response_inicial:
        all_coords = []
        for section in response_inicial.get("routes", [])[0].get("sections", []):
            polyline = section.get("polyline")
            if polyline:
                all_coords.extend(decode_polyline(polyline))

        (coordenadas_processadas,
         total_coordinates,
         coordinates_found_in_summary,
         penalties_applied,
         group_maes) = process_coordinates(
             all_coords, dia, mes, hora_, minuto, avoid_coords
         )

    # Rota final (após penalidades)
    (final_duration, final_distance, rota_final_instructions,
     payload_final, response_final) = calculate_route_with_waypoint(
         origin, destination, waypoint, avoid_coords
    )

    print(f"Duração final: {final_duration} minutos, Distância final: {final_distance} metros")
    for instr in rota_final_instructions:
        print(" -", instr)

    # Exportar relatório
    filename = generate_random_filename()
    export_all_variables_to_txt(
        filename,
        configuracao_da_rota={
            "Origem": origin,
            "Destino": destination,
            "Waypoint": waypoint or "None",
            "Data/Hora": f"{date} {hour}"
        },
        estatisticas_do_processamento={
            "Total coords extraídas": total_coordinates,
            "Coordenadas no summary": coordinates_found_in_summary,
            "Penalidades aplicadas": penalties_applied
        },
        duracao_da_rota={
            "Inicial": f"{initial_duration} min",
            "Final": f"{final_duration} min"
        },
        distancia_da_rota={
            "Inicial": f"{initial_distance} m",
            "Final": f"{final_distance} m"
        },
        instrucoes_de_rota={
            "Rota inicial": rota_inicial_instructions,
            "Rota final": rota_final_instructions
        },
        maes_por_agrupamento={
            group: f"{mae:.2f}" if mae != "N/A" else "N/A"
            for group, mae in group_maes.items()
        },
        payloads_e_respostas={
            "Payload inicial": payload_inicial,
            "Resposta inicial": response_inicial,
            "Payload final": payload_final,
            "Resposta final": response_final
        },
        rotas_completas={
            "Coords iniciais": decode_polyline(response_inicial["routes"][0]["sections"][0]["polyline"])
            if response_inicial else [],
            "Coords finais": decode_polyline(response_final["routes"][0]["sections"][0]["polyline"])
            if response_final else []
        },
        coordenadas_processadas=[
            {
                "Coordenada": (lat, lon),
                "Endereço": address or "None",
                "Grupo": group or "None",
                "Volume previsto": vol,
                "Volume médio": stats.get("mean_volume", "N/A"),
                "Erro Absoluto": abs(vol - stats.get("mean_volume", 0))
                    if stats.get("mean_volume", 0) > 0 else "N/A",
                "Evitada": "Sim" if vol > 80 else "Não",
            }
            for (lat, lon, vol, address, group, stats, graph_file) in coordenadas_processadas
        ]
    )

# =============================================================================
# INTERFACE TKINTER
# =============================================================================

def create_gui():
    """
    Função que cria a interface gráfica (GUI) para inserir dados de origem, destino e waypoint.
    Permite ao usuário definir a data/hora e então chama 'optimize_route_with_waypoint' para
    executar o fluxo principal de cálculo e processamento.
    """
    def generate_route():
        origin = entry_origin.get()
        destination = entry_destination.get()
        waypoint = entry_waypoint.get()
        date = entry_date.get()
        hour = entry_hour.get()

        if not origin or not destination or not date or not hour:
            messagebox.showerror("Erro", "Todos os campos são obrigatórios.")
            return

        waypoint_coords = None
        if waypoint:
            waypoint_coords = get_coordinates(waypoint)

        try:
            optimize_route_with_waypoint(
                origin,
                destination,
                date,
                hour,
                waypoint=waypoint_coords
            )
            messagebox.showinfo("Sucesso", "Rota gerada e resultados exportados.")
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {e}")

    root = tk.Tk()
    root.title("Otimização de Rota (HERE)")

    tk.Label(root, text="Origem:").grid(row=0, column=0, padx=5, pady=5)
    entry_origin = tk.Entry(root, width=50)
    entry_origin.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(root, text="Destino:").grid(row=1, column=0, padx=5, pady=5)
    entry_destination = tk.Entry(root, width=50)
    entry_destination.grid(row=1, column=1, padx=5, pady=5)

    tk.Label(root, text="Waypoint (Opcional):").grid(row=2, column=0, padx=5, pady=5)
    entry_waypoint = tk.Entry(root, width=50)
    entry_waypoint.grid(row=2, column=1, padx=5, pady=5)

    tk.Label(root, text="Data (YYYY-MM-DD):").grid(row=3, column=0, padx=5, pady=5)
    entry_date = tk.Entry(root, width=50)
    entry_date.grid(row=3, column=1, padx=5, pady=5)

    tk.Label(root, text="Hora (HH:MM):").grid(row=4, column=0, padx=5, pady=5)
    entry_hour = tk.Entry(root, width=50)
    entry_hour.grid(row=4, column=1, padx=5, pady=5)

    tk.Button(root, text="Gerar Rota", command=generate_route).grid(row=5, column=0, columnspan=2, pady=10)

    root.mainloop()

# Se o script for executado diretamente, inicia a GUI.
if __name__ == "__main__":
    create_gui()
