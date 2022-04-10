# merFISH Visualiser
Using the Napari multidimensional viewer for efficient viewing of merFISH results

# Installation

To use this visualiser, use conda to install the environment

```
conda env create -f environment.yaml
conda activate merfish_visualizer
```

# Usage

There is a `config.yaml` file that will take in the following parameters

```
{
    "raw_data_dir": [str],
    "analysis_dir": [str],
    "stage2pix_scaling":[float],
    "stage2z_scaling":[float],
    "decoded_img":[true or false],
    "ir_upper":[int],
    "z_lower":[int]
}
```

If `decoded_img` is `false`, then `analysis_dir` is not looked at. This would be useful for when you want to visualise the run prior to analysis.

The rest of the parameters are taken from the `stagePos_Round#1.xlsx` file that exists within the vancouver merFish runs.

The `"ir_upper"` is one plus the number of imaging rounds that you want to look at. In addition, the `"z_lower"` is the parameters for the lowest z-height you want to look at. The visualiser will assign the lowest value with an index of zero on the scroll bar. If there are certain z-slices or FOVs missing in the analysis, then the visualiser will skip loading those entirely.  

Once the `config.yaml` file has been defined
```
python main.py
```

will launch the multi-dimensional napari viewer. 

# TODO:

1. Add schema
2. Add selection of imaging rounds
3. Add widget to show codebook and select genes from it.
