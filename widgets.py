from qtpy.QtWidgets import QMainWindow, QLabel,QGridLayout
import napari
from qtpy import uic
from pathlib import Path
import pandas as pd
import dask.array as da

# Define the main window class
class FancyGUI(QMainWindow):
    def __init__(self, napari_viewer:napari.Viewer,codebook:pd.DataFrame,fovs,x_loc,y_loc,img_scale):          # include napari_viewer as argument (it has to have this name)
        super().__init__()
        self.viewer = napari_viewer
        self.fovs = fovs
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.img_scale=img_scale



        self.codebook = codebook
        self.UI_FILE = str(Path(__file__).parent / "barcode_viewer.ui")  # path to .ui file
        uic.loadUi(self.UI_FILE, self)           # load QtDesigner .ui file
        # self.widgetArrangement = QGridLayout()
        # self.widgetArrangement.addWidget(self.geneName)
        # self.widgetArrangement.addWidget(self.barcode)

        # self.setLayout(self.widgetArrangement)
        self.btnMakeLayer.clicked.connect(self.makeGeneLayer)
        self.viewer.dims.events.connect(self.updateBarcode)
        

    def updateBarcode(self):

        current_slice = self.viewer.dims.current_step

        self.lblDecodedBarcode.setText(self.codebook.iloc[current_slice[0]]['barcode'])
        self.lbDecodedGeneName.setText(self.codebook.iloc[current_slice[0]]['name'])
    

    def makeGeneLayer(self):
        
        current_slice = self.viewer.dims.current_step
        for idx,fov in enumerate(self.fovs):
            

            if idx==0:
                l = self.viewer.layers[f"decoded"]    
            else:
                try:
                    l = self.viewer.layers[f"decoded [{idx}]"]
                except KeyError:
                    continue
            
            self.viewer.add_image(
                            da.repeat(
                                da.stack([l.data[current_slice[0],:,:,:,:]],axis=0),
                            repeats = len(self.codebook),
                            axis=0),
                        name=f"Gene: {current_slice[0]} decoded [{idx}]",
                        translate=(self.x_loc[idx],-self.y_loc[idx]),
                        scale=self.img_scale,
                        )
    
        print("Layers added")
        