# FYP: Sand-Dust Image Restoration

## how to start
1. install packages in requirements.txt
2. set pythonpath to the anaconda env dir and root dir of this project
3. (download SIE dataset from Google Drive to the Data directory)

## how to name the dataset folder in Data folder
1. For a paired dataset
- Ground truth images should be named as "Ground_truth".
- Sand dust images should be named as "Sand_dust_images".
```
Data/
└── Synthetic_images/
    ├── Ground_truth/
    │   ├── ...
    │   ├── ...
    │   └── ...
    └── Sand_dust_images/
        ├── ...
        ├── ...
        └── ...
```

2. For an unpaired dataset
- Sand dust images should be named as "Sand_dust_images".
```
Data/
└── Hazed_images/
    ├── Sand_dust_images/
        ├── ...
        ├── ...
        └── ...
```