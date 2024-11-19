from point_cloud import visualize_clustered_parquet

data = 'data/clustered_points_dbscan5.parquet'

visualize_clustered_parquet(data, point_size=2, show_noise=False, color_scheme='YlOrRd')

# visualize_clustered_parquet(data, point_size=2, show_noise=True, color_scheme=['cyan'])


# # Ejemplo de uso con diferentes opciones de color:
# parquet_file = "data/clustered_points.parquet"
# point_size = 3
# sampling = 0.5

# # 1. Usar un mapa de colores de matplotlib
# visualize_clustered_parquet(parquet_file, point_size, sampling, color_scheme='magma')

# # 2. Usar una lista de colores
# visualize_clustered_parquet(parquet_file, point_size, sampling, color_scheme=['red', 'green', 'blue', 'yellow', 'cyan','magenta'])

# # 3.  Usar colores personalizados para cada etiqueta
# custom_colors = {-1: 'gray', 0: 'red', 1: 'green', 2: 'blue', 3: [1.0, 0.5, 0.2]}  # RGB o nombre de color
# visualize_clustered_parquet(parquet_file, point_size, sampling, custom_colors=custom_colors)
