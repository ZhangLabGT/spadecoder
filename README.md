### Spatiotemporal cell type deconvolution leveraging tissue structure

SpaDecoder is a a spatial spot deconvolution method leveraging individual single cell reference profiles, as opposed to cell-type aggregated, and 3D spatial tissue structure with the help of an adaptive 3D weighted spatial Gaussian kernel which enables information sharing across transcriptionally similar spatially proximal spots. We predict cell-type proportions by optimizating a matrix factorization based loss function. In order to adapt to both homogeneous as well as heterogeneous tissue environments, we develop a permutation based localized weighted spatial autocorrelation metric which when applied to each spot efficiently selects the neighborhood within which to pool transcriptomic and spatial information to improve deconvolution predictions. We probabilistically align adjacent tissue slices in 3D and infer intermediate slice expression to augment our 3D spatial tissue slice stack. Utilizing a learnable parameter, we model the batch effect between the reference scRNA-seq and the spatial dataset. We define several metrics and perform downstream analyses to distill SpaDecoder cell type proportions and scRNA-seq-spatial maps into interpretable biological findings. Our preprint can be accessed at [aioarxiv](https://www.biorxiv.org/content/10.64898/2026.02.10.705204v1)
An overview of SpaDecoder and downstream analyses is shown in the figure below. 

![Overview of SpaDecoder](images/fig1.png)


SpaDecoder uses  
0. python==3.12.3
1. scanpy==1.10.1 
2. squidpy==1.2.2
3. scipy==1.13.1
4. numpy==1.26.4
5. pandas==2.2.2
6. scikit-learn==1.5.0
7. moscot==0.4.2
8. matplotlib==3.9.0
9. seaborn==0.13.2
10. torch==2.3.1+cu121


This repo is organized as followed: 
1. spadecoder.py - main code for SpaDecoder 
2. processing_for_model.py - auxiliiary processing code 
3. post_processing.py - code for downstream metrics 
4. evaluations.py - functions for performance evaluation 
5. simulations_tissue.py - code to simulate spots and slice stacks 
6. applications/datasets - contains processing code as well as run scripts for spadecoder and other baselines 
7. applications/downstream_analyses - contains code for generating each figure in the manuscript.  


#### Citation 
If you find SpaDecoder helpful, kindly cite 
```
Spatiotemporal cell type deconvolution leveraging tissue structure
Macrina Maria Lobo, Ziqi Zhang, Xiuwei Zhang
bioRxiv 2026.02.10.705204; doi: https://doi.org/10.64898/2026.02.10.705204
```

#### Contact 
Other datasets used in the paper can be requested and GitHub issues are welcomed. For any additional questions feel free to contact Macrina Lobo (mlobo6 at gatech.edu)