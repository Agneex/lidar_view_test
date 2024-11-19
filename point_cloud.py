import os
import gc
import laspy
import numpy as np
import pandas as pd
import pyvista as pv


def load_las_data(las_file_path, block_size=1_000_000, sampling_factor=10):
    """
    Carga un archivo .las y devuelve una matriz de puntos 3D después de aplicar submuestreo.

    Parámetros:
    - las_file_path (str): Ruta al archivo .las a cargar.
    - block_size (int, opcional): Tamaño de bloque para cargar puntos de manera incremental.
                                  Valor predeterminado: 1,000,000.
    - sampling_factor (int, opcional): Factor de submuestreo para reducir el número de puntos.
                                       Valor predeterminado: 10.

    Devuelve:
    - np.ndarray: Matriz de puntos 3D después de aplicar submuestreo.
                  Si hay algún error durante la carga o el procesamiento, devuelve None.
    """

    try:
        las = laspy.read(las_file_path)
        print(f"Archivo .las cargado con éxito: {las_file_path}")

        total_points = len(las.X)
        print(f"Total de puntos en el archivo: {total_points}")

        point_data = []
        for start in range(0, total_points, block_size):
            end = min(start + block_size, total_points)
            print(f"Cargando puntos {start} a {end}...")

            indices = np.arange(start, end, sampling_factor)  # Submuestreo
            x = las.X[indices] * las.header.scale[0] + las.header.offset[0]
            y = las.Y[indices] * las.header.scale[1] + las.header.offset[1]
            z = las.Z[indices] * las.header.scale[2] + las.header.offset[2]

            block_points = np.vstack((x, y, z)).T.astype(np.float64)
            point_data.append(block_points)

            del x, y, z, indices, block_points
            gc.collect()

        point_data = np.vstack(point_data) if point_data else np.array([]) # Manejar archivos vacíos o con errores

        return point_data

    except Exception as e:
        print(f"Error al cargar el archivo .las: {e}")
        return None



def visualize_point_cloud(point_data, scalars=None, p_size=1):  # Agregar parámetro 'scalars'
    """Visualiza la nube de puntos con PyVista.

    Args:
        point_data (np.ndarray): Array de puntos (x, y, z).
        scalars (np.ndarray, optional): Array de valores escalares para colorear los puntos.
                                        Debe tener la misma longitud que point_data.
    """
    if point_data is None or not point_data.size:
        print("No hay datos para visualizar.")
        return

    try:
        num_cols = point_data.shape[1]

        if num_cols == 3:  # Solo coordenadas x, y, z
            point_cloud = pv.PolyData(point_data)
            plotter = pv.Plotter()
            plotter.add_points(point_cloud, render_points_as_spheres=True, point_size=p_size, scalars=scalars) # Usar scalars si se proporcionan
        elif num_cols == 4: # Coordenadas x,y, z y etiquetas de cluster
            point_cloud = pv.PolyData(point_data[:, :3]) # Usar las tres primeras para las coordenadas
            plotter = pv.Plotter()
            plotter.add_points(point_cloud, render_points_as_spheres=True, point_size=p_size, scalars=point_data[:, 3]) # Usar la cuarta columna para el color

        else:
            print(f"Error: El array de puntos debe tener 3 o 4 columnas. Tiene {num_cols} columnas.")
            return


        print("Iniciando visualización...")
        plotter.show()

    except Exception as e:
        print(f"Error al visualizar la nube de puntos: {e}")


def visualize_clustered_parquet(parquet_file, point_size=1, sampling_fraction=1.0, show_noise=False, color_scheme='viridis', custom_colors=None):  # Nuevos parámetros
    """Carga y visualiza datos de un archivo Parquet.

    Args:
        parquet_file (str): Ruta al archivo Parquet.
        point_size (int): Tamaño de los puntos.
        sampling_fraction (float): Fracción de puntos a mostrar (0.0 - 1.0).
        show_noise (bool): Si es True, muestra los puntos de ruido (etiqueta -1). Si es False, los filtra.
        color_scheme (str o list, opcional): Esquema de color. Puede ser un string con el nombre de un mapa de colores de matplotlib (ej. 'viridis', 'jet', 'magma', etc.) o una lista de colores en formato RGB o hexadecimal. Por defecto es 'viridis'.
        custom_colors (dict, opcional): Diccionario que mapea las etiquetas de cluster a colores específicos.  Por ejemplo:  `{-1: 'red', 0: 'blue', 1: 'green'}`. Si se proporciona, anula `color_scheme`.

    """
    try:
        df = pd.read_parquet(parquet_file, engine='pyarrow')

        if not all(col in df.columns for col in ['x', 'y', 'z', 'cluster_label']):
            raise ValueError("El archivo Parquet debe contener 'x', 'y', 'z' y 'cluster_label'.")

        if not show_noise:
            df = df[df['cluster_label'] != -1]

        if sampling_fraction < 1.0:
            sampled_df = df.sample(frac=sampling_fraction)
            point_data = sampled_df[['x', 'y', 'z']].to_numpy()
            cluster_labels = sampled_df['cluster_label'].to_numpy()
        else:
            point_data = df[['x', 'y', 'z']].to_numpy()
            cluster_labels = df['cluster_label'].to_numpy()


        point_cloud = pv.PolyData(point_data)
        point_cloud["cluster_label"] = cluster_labels

        plotter = pv.Plotter()
        plotter.background_color = 'gray'



        if custom_colors:
            # Usar colores personalizados si se proporcionan
            plotter.add_mesh(point_cloud, scalars="cluster_label", cmap=custom_colors, point_size=point_size, render_points_as_spheres=True)

        elif isinstance(color_scheme, list):
             # Usar la lista de colores si se proporciona.
            plotter.add_mesh(point_cloud, scalars="cluster_label", cmap=color_scheme, point_size=point_size, render_points_as_spheres=True)
        else:
            # Usar el mapa de colores especificado
            plotter.add_mesh(point_cloud, scalars="cluster_label", cmap=color_scheme, point_size=point_size, render_points_as_spheres=True)



        plotter.show_grid()
        plotter.show()



    except FileNotFoundError:
        print(f"Error: Archivo Parquet no encontrado: {parquet_file}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error al visualizar: {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    las_file_path = os.environ["LAS_FILE_PATH"]

    try:
        point_data = load_las_data(las_file_path, block_size=1_000_000, sampling_factor=10)

        if point_data is not None:
            print(f"Nube de puntos cargada con {point_data.shape[0]} puntos después del submuestreo.")
            visualize_point_cloud(point_data)


    except Exception as e:
        print(f"Error general: {e}")

    finally:
        gc.collect()
        print("Memoria liberada correctamente.")