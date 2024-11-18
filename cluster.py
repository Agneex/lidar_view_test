import os
import cudf
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from cuml.cluster import DBSCAN
from point_cloud import load_las_data, visualize_point_cloud

load_dotenv()

las_file_path = os.environ["LAS_FILE_PATH"]

point_data = load_las_data(las_file_path, block_size=5_000_000, sampling_factor=1)

if point_data is None:
    print("Error cargando datos LAS. Saliendo.")
    exit()

point_data = point_data[0:1000000]

# Número de clusters para KMeans
n_clusters = 6  # Ajustar según sea necesario

# Convertir los datos a cudf DataFrame (necesario para CuML)
try:
    point_data_cudf = cudf.DataFrame(point_data) # Conversión directa
except Exception as e:  # Manejar errores si la conversión falla
    print(f"Error al convertir a cudf DataFrame: {e}")
    exit()



# Aplicar clustering KMeans con CuML
dbscan = DBSCAN(eps=0.5, min_samples=n_clusters)
labels_cudf = dbscan.fit_predict(point_data_cudf)


# Convertir las etiquetas de vuelta a numpy array (opcional, si se necesita)
labels = labels_cudf.to_numpy()


# Agregar las etiquetas a los datos de la nube de puntos
clustered_points = np.column_stack((point_data, labels))


# Convertir a DataFrame de Pandas (opcional)
df_clustered = pd.DataFrame(clustered_points, columns=['x', 'y', 'z', 'cluster_label'])

# Imprimir las primeras filas del DataFrame
print(df_clustered.head())

visualize_point_cloud(df_clustered.to_numpy())

df_clustered.to_parquet("data/clustered_points.parquet", index=False)
