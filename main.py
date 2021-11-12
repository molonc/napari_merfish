import napari
import skimage.io as skio
import numpy as np
import pandas as pd
from dask_image.imread import imread
import dask.array as da
from dask import delayed
import json

if __name__=="__main__":
    config = json.load(open('config.json'))
    ir_upper=9 #Has to 9 or less
    raw_data_dir = config["raw_data_dir"]
    analysis_dir = config["analysis_dir"]
    stagepos_file = raw_data_dir + 'stagePos_Round#1.xlsx'

    stage = pd.read_excel(stagepos_file)
    z_lower = 2
    z_num = len(stage.columns)-5
    fovs = stage['Var1_ 1'].apply(lambda x: f'{x:03d}')
    x_loc = stage['Var1_ 4']
    y_loc = stage['Var1_ 5']
    z_spacing = np.abs(stage['Var1_ 6'][0]-stage['Var1_ 7'][0])

    x_loc = x_loc-x_loc.iloc[0]
    y_loc = y_loc-y_loc.iloc[0]

    stage2pix_scaling=config["stage2pix_scaling"] # nikon/pix
    stage2z_scaling = config["stage2pix_scaling"] #nikon/z
    
    viewer = napari.Viewer()
    for idx,fov in enumerate(fovs):
        
        pattern_img =  'merFISH_{:02d}_' +f'{fov}_*.TIFF'
        image_root = f'/decoding/decoded_images/decoded_{fov}'+'_{:02d}.npy'

        
        if config["decoded_img"]:
            
            sample = np.load(analysis_dir+image_root.format(z_lower))

            lazy_npload = delayed(np.load)
            _zs = np.arange(z_lower,z_num+1)
            lazy_decodes = [lazy_npload(analysis_dir+image_root.format(z)) for z in _zs]

            dask_arrays = [
            da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
            for delayed_reader in lazy_decodes
            ]
            stack = da.stack(dask_arrays, axis=0)
            print(stack.shape)
            #stack = da.broadcast_to(stack,(1,stack.shape[0],stack.shape[1],stack.shape[2]))
            #add decoded image
            viewer.add_image(stack,translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),name=f'decoded',scale=[1,z_spacing/stage2z_scaling,1,1])

        #add 473 volume
        channel = 473
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),name=f'fov:{fov}, {channel}nm volume',opacity=0.5,scale=[1,z_spacing/stage2z_scaling,1,1]) 
        #add 647 volume 
        channel = 647
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),name=f'fov:{fov}, {channel}nm volume',opacity=0.5,colormap='red',scale=[1,z_spacing/stage2z_scaling,1,1]) 
        #add 750 volume 
        channel = 750
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),name=f'fov:{fov}, {channel}nm volume',opacity=0.5,colormap='green',scale=[1,z_spacing/stage2z_scaling,1,1]) 

        viewer.dims.axis_labels = ['IR', 'Z', 'Y', 'X']



    viewer.reset_view()# start the event loop and show the viewer
    napari.run()
