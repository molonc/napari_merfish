import napari
import skimage.io as skio
import numpy as np
import pandas as pd
from dask_image.imread import imread
import dask.array as da
from dask import delayed
import json
import os

def gene_mask(arr,g):
    v = da.equal(arr,g)
    return v

def dask_reshape(arr,gmax):
    
    gene_map = [gene_mask(arr-1,g) for g in range(gmax)]
    x=da.stack(1*gene_map,axis=0)
    return x
    

if __name__=="__main__":
    config = json.load(open('config.json'))
    ir_upper=config["ir_upper"] #Has to 9 or less
    raw_data_dir = config["raw_data_dir"]
    analysis_dir = config["analysis_dir"]
    stagepos_file = os.path.join(raw_data_dir,'stagePos_Round#1.xlsx')

    stage = pd.read_excel(stagepos_file)
    z_lower = config["z_lower"]
    z_num = len(stage.columns)-5
    if 'Var1_ 1' in stage.columns: #Backwards compatability with old stagepos files
        fovs = stage['Var1_ 1'].apply(lambda x: f'{x:03d}')
        x_loc = stage['Var1_ 4']
        y_loc = stage['Var1_ 5']
        z_spacing = np.abs(stage['Var1_ 6'][0]-stage['Var1_ 7'][0])
    else:
        fovs = stage['tile_number'].apply(lambda x: f'{x:03d}')
        x_loc = stage['stage_pos_x']
        y_loc = stage['stage_pos_y']
        z_spacing = np.abs(stage['z_position_1'][0]-stage['z_position_2'][0])
    
    x_loc = x_loc-x_loc.iloc[0]
    y_loc = y_loc-y_loc.iloc[0]

    stage2pix_scaling=config["stage2pix_scaling"] # nikon/pix
    stage2z_scaling = config["stage2pix_scaling"] #nikon/z
    
    viewer = napari.Viewer()
    for idx,fov in enumerate(fovs):
        
        pattern_img =  'merFISH_{:02d}_' +f'{fov}_*.TIFF'
        image_root = f'/decoding/decoded_images/decoded_{fov}'+'_{:02d}.npy'

        
        if config["decoded_img"]:
            name =analysis_dir+image_root.format(z_lower)
            if not os.path.isfile(name):
                print(f"{name} does not exist")
                continue
            sample = np.load(name)
            num_genes = sample.max()
            
            lazy_npload = delayed(np.load)
            lazy_reshapefn = delayed(dask_reshape)
            _zs = np.arange(z_lower,z_num+1)
            lazy_decodes = [lazy_npload(analysis_dir+image_root.format(z)) for z in _zs]
            
            dask_arrays = [
            da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
            for delayed_reader in lazy_decodes
            ]
            lazy_reshapes = [lazy_reshapefn(da,num_genes) for da in dask_arrays]
            dask_arrays = [
            da.from_delayed(lr, shape=(num_genes,*sample.shape), dtype=sample.dtype)
            for lr in lazy_reshapes
            ]

            stack = da.stack(dask_arrays, axis=0)
            
            stack = da.broadcast_to(stack,(1,stack.shape[0],stack.shape[1],stack.shape[2],stack.shape[3]))
            stack = da.transpose(stack,[2,0,1,3,4])
            #add decoded image
            viewer.add_image(stack,translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),name=f'decoded',scale=[1,1,z_spacing/stage2z_scaling,1,1])

        #add 473 volume
        channel = 473
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,
                                translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),
                                name=f'fov:{fov}, {channel}nm volume',
                                opacity=0.5,
                                scale=[1,z_spacing/stage2z_scaling,1,1],
                                contrast_limits=[0,2**16]) 
        #add 561 volume
        channel = 561
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,
                                translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),
                                name=f'fov:{fov}, {channel}nm volume',
                                opacity=0.5,
                                scale=[1,z_spacing/stage2z_scaling,1,1],
                                contrast_limits=[0,2**16])         
        #add 647 volume 
        channel = 647
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,
                                translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),
                                name=f'fov:{fov}, {channel}nm volume',
                                opacity=0.5,
                                scale=[1,z_spacing/stage2z_scaling,1,1],
                                contrast_limits=[0,2**16],
                                colormap='yellow')
        #add 750 volume 
        channel = 750
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,
                                translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),
                                name=f'fov:{fov}, {channel}nm volume',
                                opacity=0.5,
                                scale=[1,z_spacing/stage2z_scaling,1,1],
                                contrast_limits=[0,2**16],
                                colormap='green')
        viewer.dims.axis_labels = ['GN','IR', 'Z', 'Y', 'X']



    viewer.reset_view()# start the event loop and show the viewer
    napari.run()
