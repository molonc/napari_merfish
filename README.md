# merFISH Visualiser
Using the Napari multidimensional viewer for efficient viewing of merFISH results

# Installation

To use this visualiser, use conda to install the environment

```
conda create env -f environment.yaml
conda activate merfish_visualizer
```

# Usage

There is a `config.yaml` file that will take in the following parameters

```
{
    "raw_data_dir": "/Volumes/MERFISH_COLD/XP872/20210917 4T1 C1E1/4T1/C1E1/",
    "analysis_dir": "/Volumes/MERFISH_COLD/XP872/20210917 4T1 C1E1/Analyzed_z_slim_test_at0",
    "stage2pix_scaling":0.17525,
    "stage2z_scaling":1,
    "decoded_img":true
}
```

If `decoded_img` is `false`, then `analysis_dir` is not looked at. This would be useful for when you want to visualise the run prior to analysis.

The rest of the parameters are taken from the `stagePos_Round#1.xlsx` file that exists within the vancouver merFish runs.

Once the `config.yaml` file has been defined
```
python main.py
```

will launch the multi-dimensional napari viewer. 