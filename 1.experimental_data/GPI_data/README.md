# Gas puff imaging data 
This folder contains the gas puff imaging data used for the paper Y. Wang et al, "Comparison of filament properties in real-size GBS simulations and experiments of TCV-X21". It is an extension of the TCV-X21 dataset. 

It includes 3 data files under `/GPI_TCVX21`, with the information listed below


| File Name | Position | Time window for filament tracking [s]| Spatial resolution (pixels)|
|:-----|:-----|:-----|:-----|
| `/GPI_TCVX21/77028_1.05_1.054.mat` | Outboard Midplane    | 1.05-1.054    | 12*10|
| `/GPI_TCVX21/70336_1.5750_1.5854.mat`| X-point region     | 1.58-1.5804    |96*128|
| `/GPI_TCVX21/70545_1.5650_1.5790.mat`| Divertor leg     | 1.57-1.574    |96*128|


The datafiles are named with the discharge number and the starting and ending time, then the filament tracking algorithm used part of the data. 

A jupyter notebook format script that loads and preview the data file is given as `data_processing.py`.

Detailed information about the diagnostic and the dataset can be found in the paper.
The full time range of the data in the gas puff can be obtained by contacting the authors of the paper. yinghan.wang@epfl.ch 
