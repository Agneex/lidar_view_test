import matplotlib.pyplot as plt
from whitebox_workflows import show, WbEnvironment

wbe = WbEnvironment()
wbe.working_directory = "/media/oscar/data/Interventoria_IA/02_LiDAR/ML-5FFCAA-2024-10-12-13-24-37/clouds"
lidar = wbe.read_lidar('ML-5FFCAA-2024-10-12-13-24-37.las')

# show(lidar,
#      figsize=(10,8),
#      skip=500,
#      vert_exaggeration=5.0,
#      marker='o',
#      s=1,
#      cmap='viridis',
#      colorbar_kwargs={'location': 'right',
#                       'shrink': 0.5,
#                       'label': 'Elevation (m)',
#                       'pad': 0.1})

# Let's create our own custom axes (ax) to pass to the show function
fig = plt.figure()
fig.set_dpi(180.0) # Let's make this figure higher resolution.
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_axis_off() # This line ensures that axes will not be rendered

show(lidar,
     ax=ax, # pass our axes here
     figsize=(15, 13),
     skip=2000, # and plot with higher point density
     vert_exaggeration=5.0,
     marker='o',
     s=0.25, # higher point density might need smaller points
     cmap='viridis',
     colorbar_kwargs={'location': 'right',  'shrink': 0.3, 'label': 'Elevation (m)', 'pad': 0.0})

plt.show()
