from point_cloud import visualize_clustered_parquet

data = 'data/clustered_points.parquet'

visualize_clustered_parquet(data, p_size=2, show_noise=True)
