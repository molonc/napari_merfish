import napari
import skimage.io as skio
import numpy as np
import pandas as pd
from dask_image.imread import imread
import dask.array as da
if __name__=="__main__":
    
    _z = 1
    ir_upper=9 #Has to 9 or less
    raw_data_dir = '/Volumes/MERFISH_COLD/XP872/20210917 4T1 C1E1/4T1/C1E1/'
    
    stagepos_file = raw_data_dir + 'stagePos_Round#1.xlsx'

    stage = pd.read_excel(stagepos_file)
    z_num = len(stage.columns)-5
    fovs = stage['Var1_ 1'].apply(lambda x: f'{x:03d}')
    x_loc = stage['Var1_ 4']
    y_loc = stage['Var1_ 5']
    z_spacing = np.abs(stage['Var1_ 6'][0]-stage['Var1_ 7'][0])

    x_loc = x_loc-x_loc.iloc[0]
    y_loc = y_loc-y_loc.iloc[0]

    stage2pix_scaling=0.17525 # nikon/pix
    stage2z_scaling = 1 #nikon/z
    analysis_dir = '/Volumes/MERFISH_COLD/XP872/20210917 4T1 C1E1/Analyzed_02_at4'
    viewer = napari.Viewer()
    for idx,fov in enumerate(fovs):
        
        pattern_img =  'merFISH_{:02d}_' +f'{fov}_*.TIFF'
        image_root = f'/decoding/decoded_images/decoded_{fov}.npy'

        img = np.load(analysis_dir+image_root)

        #add decoded image
        viewer.add_image(img,translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),name=f'decoded')

        #add 473 volume
        channel = 473
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),name=f'473 volume fov:{fov}',opacity=0.5,scale=[1,z_spacing/stage2z_scaling,1,1]) 
        #add 647 volume 
        channel = 647
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),name=f'647 volume fov:{fov}',opacity=0.5,colormap='red',scale=[1,z_spacing/stage2z_scaling,1,1]) 
        #add 750 volume 
        channel = 750
        channel_format = f'{channel}nm, Raw/'
        irs = [imread(raw_data_dir + channel_format + pattern_img.format(ir)) for ir in range(1,ir_upper)]
        stack = da.stack(irs)    
        viewer.add_image(stack,translate=(x_loc.iloc[idx]/stage2pix_scaling,y_loc[idx]/stage2pix_scaling),name=f'750 volume fov:{fov}',opacity=0.5,colormap='green',scale=[1,z_spacing/stage2z_scaling,1,1]) 

        viewer.dims.axis_labels = ['IR', 'Z', 'Y', 'X']



    # start the event loop and show the viewer
    napari.run()