import os
import cudf
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from cuml.cluster import DBSCAN  # Importar DBSCAN de CuML
from point_cloud import load_las_data, visualize_point_cloud

load_dotenv()

las_file_path = os.environ["LAS_FILE_PATH"]

point_data = load_las_data(las_file_path, block_size=5_000_000, sampling_factor=1)

if point_data is None:
    print("Error cargando datos LAS. Saliendo.")
    exit()

# Selecciona la porción de datos que quieres usar (ajusta según tus necesidades y la memoria disponible)
point_data = point_data[20000000:21000000]

try:
    point_data_cudf = cudf.DataFrame(point_data)
except Exception as e:
    print(f"Error al convertir a cudf DataFrame: {e}")
    exit()


# Crea una instancia de DBSCAN. Ajusta eps y min_samples según tus datos.
eps = 1.45  # Radio de vecindad
min_samples = 1300  # Número mínimo de puntos para formar un cluster

dbscan = DBSCAN(eps=eps,
                min_samples=min_samples,
                max_mbytes_per_batch=8000)

# Ajusta el modelo a todos los datos y obtén las etiquetas
labels_cudf = dbscan.fit_predict(point_data_cudf)

# Convierte las etiquetas a un array NumPy
labels = labels_cudf.to_numpy()

# Crea el DataFrame de Pandas
clustered_points = np.column_stack((point_data, labels))
df_clustered = pd.DataFrame(clustered_points, columns=['x', 'y', 'z', 'cluster_label'])
df_clustered['cluster_label'] = df_clustered['cluster_label'].astype(int)

print(df_clustered.head())

visualize_point_cloud(df_clustered.to_numpy())

df_clustered.to_parquet("data/clustered_points_dbscan.parquet", index=False)
