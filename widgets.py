from qtpy.QtWidgets import QMainWindow, QLabel,QGridLayout
import napari
from qtpy import uic
from pathlib import Path
import pandas as pd

# Define the main window class
class FancyGUI(QMainWindow):
    def __init__(self, napari_viewer:napari.Viewer,codebook:pd.DataFrame):          # include napari_viewer as argument (it has to have this name)
        super().__init__()
        self.viewer = napari_viewer
        
        self.codebook = codebook
        self.UI_FILE = str(Path(__file__).parent / "barcode_viewer.ui")  # path to .ui file
        uic.loadUi(self.UI_FILE, self)           # load QtDesigner .ui file
        # self.widgetArrangement = QGridLayout()
        # self.widgetArrangement.addWidget(self.geneName)
        # self.widgetArrangement.addWidget(self.barcode)

        # self.setLayout(self.widgetArrangement)
        self.viewer.dims.events.connect(self.updateBarcode)
        

    def updateBarcode(self):

        current_slice = self.viewer.dims.current_step

        self.lblDecodedBarcode.setText(self.codebook.iloc[current_slice[0]]['barcode'])
        self.lbDecodedGeneName.setText(self.codebook.iloc[current_slice[0]]['name'])
        