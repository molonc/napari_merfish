from magicgui import magicgui
import napari
import skimage.io as skio
import scipy.ndimage as ndi
import numpy as np
import pandas as pd
from dask_image.imread import imread
from dask_image.ndfourier import fourier_shift
import dask.array as da
from dask.array.image import imread as daimread
from dask import delayed
import json
import os
import functools
import pandas as pd
import matplotlib.pyplot as plt
from napari.types import ImageData, PointsData
import widgets
from magicgui import magicgui
import functools

def gene_mask(arr,g):
    v = da.equal(arr,g)
    return v

def dask_reshape(arr,gmax):
    
    gene_map = [gene_mask(arr-1,g) for g in range(gmax)]
    x=da.stack(1*gene_map,axis=0)
    return x
    
def getTLBF(df:pd.DataFrame,im_shape):
    # TODO: THIS NEEDS TO BE CHANGED WITH THE NEW CSV FORMAT
    shift_r = [0]
    shift_r.extend(df['shift_x'].to_list())
    shift_c = [0]
    shift_c.extend(df['shift_y'].to_list())

    dx_tl = np.max(np.maximum(shift_c,0))
    dy_tl = np.max(np.maximum(shift_r,0))

    dx_br = np.min(np.minimum(shift_c,0))
    dy_br = np.min(np.minimum(shift_r,0))

    tl = np.ceil([dy_tl,dx_tl]).astype(int)
    br = np.floor([im_shape[0]+dy_br,im_shape[1]+dx_br]).astype(int)
    return shift_r,shift_c,tl, br

def load_img_and_shift(fn,shift,tl,br):
    _img = skio.imread(fn)
    #just perform the shift and then trim

    shifted_img = ndi.shift(_img,shift)

    cropped_img = shifted_img#[tl[0]:br[0],tl[1]:br[1]]

    return cropped_img

if __name__=="__main__":
    config = json.load(open('config.json'))
    ir_upper=config["ir_upper"] #Has to 9 or less
    raw_data_dir = config["raw_data_dir"]
    analysis_dir = config["analysis_dir"]
    stagepos_file = os.path.join(raw_data_dir,'stagePos_Round#1.xlsx')

    if os.path.isfile(stagepos_file):
        stage = pd.read_excel(stagepos_file)
    else:
        stagepos_file = os.path.join(raw_data_dir,'stage_position_1.csv')
        stage = pd.read_csv(stagepos_file)
        
    
    z_lower = config["z_lower"]
    z_num = config["z_upper"]#len(stage.columns)-5
    _fovs = [i for i in range(1,len(stage))]
    fovs = list(map(lambda x: f'{x:03d}',_fovs))
    if 'Var1_ 1' in stage.columns: #Backwards compatability with old stagepos files
        x_loc = stage['Var1_ 5']
        y_loc = stage['Var1_ 4']
        z_spacing = np.abs(stage['Var1_ 6'][0]-stage['Var1_ 7'][0])
    elif 'x_pos' in stage.columns:
        x_loc = stage['y_pos']
        y_loc = stage['x_pos']
        z_spacing = np.abs(stage['z_slice_0'][0]-stage['z_slice_1'][0])
    else:
        x_loc = stage['stage_pos_y']
        y_loc = stage['stage_pos_x']
        z_spacing = np.abs(stage['z_position_1'][0]-stage['z_position_2'][0])
    
    x_loc = x_loc-x_loc.iloc[0]
    y_loc = y_loc-y_loc.iloc[0]

    stage2pix_scaling=config["stage2pix_scaling"] # nikon/pix
    stage2z_scaling = config["stage2pix_scaling"] #nikon/z
    
    viewer = napari.Viewer()
    for idx,fov in enumerate(fovs):
        shift_r=np.full((ir_upper,1),0)
        shift_c=np.full((ir_upper,1),0)
        tl,br = 0,0 
        pattern_img =  config["file_pattern"].format(fov=fov)
        image_root = f'/decoding/decoded_images/decoded_{fov}'+'_{:02d}.npy'
        alignment_root = f'/aligned/shift_{fov}.csv'
        
        if config["decoded_img"]:
            shift_name =analysis_dir+alignment_root
            
            codebook_name = os.path.join(config["raw_data_dir"],config["codebook_name"])
            codebook_df = pd.read_csv(codebook_name,skiprows=3)
            codebook_df=codebook_df.rename(columns={c:c.strip() for c in codebook_df.columns},errors='raise')
            name =analysis_dir+image_root.format(z_lower)
            if not os.path.isfile(name):
                print(f"{name} does not exist")
                continue
            sample = np.load(name)
            if os.path.isfile(shift_name):
                df= pd.read_csv(shift_name)
                shift_r,shift_c,tl, br = getTLBF(df,sample.shape)
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

            viewer.add_image(stack,translate=((x_loc.iloc[idx])/stage2pix_scaling,(-y_loc[idx])/stage2pix_scaling),name=f'decoded',scale=[1,1,z_spacing/stage2z_scaling,-1,1])

        #add 473 volume
        channel = 488
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir=ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,
                                translate=((x_loc[idx])/stage2pix_scaling,
                                            (-y_loc[idx])/stage2pix_scaling),
                                name=f'fov:{fov}, {channel}nm volume',
                                opacity=0.5,
                                scale=[1,z_spacing/stage2z_scaling,-1,1],
                                contrast_limits=[0,2**16]) 
        #add 561 volume
        channel = 561
        channel_format = f'{channel}nm, Raw/'
        irs = [
            daimread(raw_data_dir + channel_format + pattern_img.format(ir=ir),functools.partial(load_img_and_shift,
                                                                                                                shift = (shift_r[ir-1],shift_c[ir-1]),tl=tl,br=br
                                                                                                )
                    ) 
            for ir in range(1,ir_upper)
            ]
        # irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,
                                translate=(
                                            x_loc.iloc[idx]/stage2pix_scaling,
                                            -y_loc[idx]/stage2pix_scaling),
                                name=f'fov:{fov}, {channel}nm volume',
                                opacity=0.5,
                                scale=[1,z_spacing/stage2z_scaling,-1,1],
                                contrast_limits=[0,2**16])         
        #add 647 volume 
        channel = 647
        channel_format = f'{channel}nm, Raw/'
        irs = [
            daimread(raw_data_dir + channel_format + pattern_img.format(ir=ir),functools.partial(load_img_and_shift,
                                                                                                                shift = (shift_r[ir-1],shift_c[ir-1]),tl=tl,br=br
                                                                                                )
                    ) 
            for ir in range(1,ir_upper)
            ]
        stack = da.stack(irs)    
        viewer.add_image(stack,
                                translate=(x_loc.iloc[idx]/stage2pix_scaling,-y_loc[idx]/stage2pix_scaling),
                                name=f'fov:{fov}, {channel}nm volume',
                                opacity=0.5,
                                scale=[1,z_spacing/stage2z_scaling,-1,1],
                                contrast_limits=[0,2**16],
                                colormap='yellow')
        #add 750 volume 
        channel = 750
        channel_format = f'{channel}nm, Raw/'
        irs = [
            daimread(raw_data_dir + channel_format + pattern_img.format(ir=ir),functools.partial(load_img_and_shift,
                                                                                                                shift = (shift_r[ir-1],shift_c[ir-1]),tl=tl,br=br
                                                                                                )
                    ) 
            for ir in range(1,ir_upper)
            ]
        stack = da.stack(irs)    
        viewer.add_image(stack,
                                translate=(x_loc.iloc[idx]/stage2pix_scaling,-y_loc[idx]/stage2pix_scaling),
                                name=f'fov:{fov}, {channel}nm volume',
                                opacity=0.5,
                                scale=[1,z_spacing/stage2z_scaling,-1,1],
                                contrast_limits=[0,2**16],
                                colormap='green')
        viewer.dims.axis_labels = ['GN','IR', 'Z', 'Y', 'X']



    if config["decoded_img"]:
        barcode_viewer=widgets.FancyGUI(viewer,codebook_df,fovs=fovs,x_loc=x_loc.to_numpy()/stage2pix_scaling,y_loc=y_loc.to_numpy()/stage2pix_scaling,img_scale=[1,z_spacing/stage2z_scaling,-1,1])
        viewer.window.add_dock_widget(barcode_viewer,area='right')

    viewer.reset_view()# start the event loop and show the viewer
    napari.run()
