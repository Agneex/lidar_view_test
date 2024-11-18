import os
import cudf
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from cuml.cluster import DBSCAN
from point_cloud import load_las_data, visualize_point_cloud

load_dotenv()

las_file_path = os.environ["LAS_FILE_PATH"]

point_data = load_las_data(las_file_path, block_size=5_000_000, sampling_factor=1)

if point_data is None:
    print("Error cargando datos LAS. Saliendo.")
    exit()

point_data = point_data[20000000:21000000]

try:
    point_data_cudf = cudf.DataFrame(point_data) # Conversión directa
except Exception as e:  # Manejar errores si la conversión falla
    print(f"Error al convertir a cudf DataFrame: {e}")
    exit()

eps = 1.7
min_samples = 1000

block_size = 200_000 # Ajusta según tu memoria y el tamaño del dataset

all_labels = []

for start in tqdm(range(0, len(point_data_cudf), block_size), desc="Clustering"):
    end = min(start + block_size, len(point_data_cudf))
    block_data = point_data_cudf.iloc[start:end]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Reinicializar DBSCAN para cada bloque (opcional, pero puede ser más estable)
    block_labels = dbscan.fit_predict(block_data)

    all_labels.append(block_labels)


labels_cudf = cudf.concat(all_labels)
labels = labels_cudf.to_numpy()

clustered_points = np.column_stack((point_data, labels))

df_clustered = pd.DataFrame(clustered_points, columns=['x', 'y', 'z', 'cluster_label'])
df_clustered['cluster_label'] = df_clustered['cluster_label'].astype(int)

print(df_clustered.head())

visualize_point_cloud(df_clustered.to_numpy())

df_clustered.to_parquet("data/clustered_points.parquet", index=False)
