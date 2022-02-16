All the train/val/test splitings are the same as the original splitings in CASIA-SURF and CeFA datasets.
We sample 3 frames for most videos.

There are two small issues/sepcial settings in the protocols, and we mark here.

--------------------------------

# For the CeFA dataset:
We only sample the first frame of most of the videos in 'CeFA-Mask/3D-Mask' for testing as there are many frames missing in these directors.

---------------------------------

# For the WMCA dataset: 
the names/index of two videos are not matched in three modality

WMCA-Image/R_CDIT/100_03_015_2_10/color/0012.jpg   --> Depth  and  IR   's index not matched.
WMCA-Image/R_CDIT/101_06_067_1_02/color/0020.jpg   --> Depth  and  IR   's index not matched.
