import numpy as np
import laspy
import pyvista as pv
import gc

las_file_path = '/media/oscar/data/Interventoria_IA/02_LiDAR/ML-5FFCAA-2024-10-12-13-24-37/clouds/ML-5FFCAA-2024-10-12-13-24-37.las'


def load_las_in_blocks(las_file_path, block_size=1_000_000, sampling_factor=100):
    las = laspy.read(las_file_path)
    print(f"Archivo .las cargado con éxito: {las_file_path}")

    total_points = len(las.X)
    print(f"Total de puntos en el archivo: {total_points}")

    point_data = []
    for start in range(0, total_points, block_size):
        end = min(start + block_size, total_points)
        print(f"Cargando puntos {start} a {end}...")

        indices = np.arange(start, end, sampling_factor)
        x = (las.X[indices] * las.header.scale[0]) + las.header.offset[0]
        y = (las.Y[indices] * las.header.scale[1]) + las.header.offset[1]
        z = (las.Z[indices] * las.header.scale[2]) + las.header.offset[2]

        block_points = np.vstack((x, y, z)).T.astype(np.float64)
        point_data.append(block_points)

        del x, y, z, indices, block_points
        gc.collect()

    point_data = np.vstack(point_data)
    return point_data


try:
    point_data = load_las_in_blocks(las_file_path, block_size=1_000_000, sampling_factor=10)
    print(f"Nube de puntos cargada con {point_data.shape[0]} puntos después del submuestreo.")

    point_cloud = pv.PolyData(point_data)

    plotter = pv.Plotter()
    plotter.add_points(point_cloud, render_points_as_spheres=True, point_size=2)
    print("Iniciando visualización optimizada con PyVista...")
    plotter.show()

except Exception as e:
    print(f"Error al procesar el archivo .las: {e}")

finally:
    gc.collect()
    print("Memoria liberada correctamente.")
