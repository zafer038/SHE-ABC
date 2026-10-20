# Data

Place the ISPRS Potsdam patch set here as `data/patches/Images/Image_<k>.tif` and `data/patches/Labels/Label_<k>.tif`
(k = 0 … 2399; 300 × 300 pixels, near-infrared/red/green). The patches are cut without overlap from six tiles of the
ISPRS 2D Semantic Labeling Potsdam benchmark (https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx),
400 per tile in raster order, so patch k comes from tile k // 400.
