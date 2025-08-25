import numpy as np
import argparse
import matplotlib.pyplot as plt
import oiffile
import pyvista as pv
from scipy.ndimage import gaussian_filter
pv.set_plot_theme('dark')

parser = argparse.ArgumentParser()
parser.add_argument('-o','--oibfile', help='oib file')
parser.add_argument('--savemovie', nargs='?', const=1, type=bool, default=False)
parser.add_argument('--logscale', nargs='?', const=1, type=bool, default=False)
args = parser.parse_args()

axes = (3, 1, 0) # OIB files have axes in order Z, Y, X
with oiffile.OifFile(args.oibfile) as oib:    
    resolutions = []
    for d in axes:
        start = oib.mainfile['Axis %d Parameters Common' % d]['StartPosition']
        stop = oib.mainfile['Axis %d Parameters Common' % d]['EndPosition']
        size = oib.mainfile['Axis %d Parameters Common' % d]['MaxSize']
        resolutions.append((stop - start) / size)
    data = oib.asarray()
    data = data[0]
    print('Data shape:', data.shape)
    
cmin, cmax = data.min(), data.max()
title = 'Intensity'
    
xyres, zres = resolutions[1], resolutions[0]

scalar_bar_args={'title':title, 'n_labels':2,
                'title_font_size':30, 'label_font_size':20, 
                'fmt':'%2.1f', 'vertical':False, 
                'height':0.05, 'width':0.5, 'position_x':0.05, 'position_y':0.05}

pl = pv.Plotter(window_size=(1200, 1200))

vol = pl.add_volume(data, 
                cmap='magma',
                clim=[cmin, cmax], 
                resolution=[zres, xyres, xyres], 
                opacity='linear',
                shade=True,
                blending='composite',
                culling='back',
                show_scalar_bar=True,
                scalar_bar_args=scalar_bar_args,
                specular_power=100,
                specular=0.2,
                diffuse=0.9,
                log_scale=args.logscale
             )
vol.prop.interpolation_type = 'linear'

if args.savemovie:
    path = pl.generate_orbital_path(factor=1.5, shift=30, n_points=360)
    pl.open_movie('%s.mov' % args.oibfile.split('.oib')[0])
    pl.orbit_on_path(path, write_frames=True, step=0.05, progress_bar=True)
    pl.close()
else:
    pl.camera.zoom(1.5)
    pl.show(auto_close=False)
#conda activate vessels
#ls -lh E18.5-4x-brain-tile-1.oib
#python visualize.py -o E18.5-4x-brain-tile-1.oib