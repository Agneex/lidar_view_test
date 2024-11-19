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

point_data = point_data[20000000:21000000]

try:
    point_data_cudf = cudf.DataFrame(point_data)
except Exception as e:
    print(f"Error al convertir a cudf DataFrame: {e}")
    exit()


eps = 0.45
min_samples = 700

dbscan = DBSCAN(eps=eps,
                min_samples=min_samples,
                max_mbytes_per_batch=8000)

labels_cudf = dbscan.fit_predict(point_data_cudf)

labels = labels_cudf.to_numpy()

clustered_points = np.column_stack((point_data, labels))
df_clustered = pd.DataFrame(clustered_points, columns=['x', 'y', 'z', 'cluster_label'])
df_clustered['cluster_label'] = df_clustered['cluster_label'].astype(int)

print(df_clustered.head())

visualize_point_cloud(df_clustered.to_numpy())

df_clustered.to_parquet("data/clustered_points_dbscan7.parquet", index=False)
