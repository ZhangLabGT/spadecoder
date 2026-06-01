

# Model Card for SpaDecoder

SpaDecoder is a multi-slice spatiotemporal spot based spatial transcriptomics deconvolution tool utilizing 3D tissue structure



## Model Details

### Model Description

SpaDecoder is a a spatial spot deconvolution method leveraging individual single cell reference profiles, as opposed to cell-type aggregated, and 3D spatial tissue structure with the help of an adaptive 3D weighted spatial Gaussian kernel which enables information sharing across transcriptionally similar spatially proximal spots. We predict cell-type proportions by optimizating a matrix factorization based loss function. In order to adapt to both homogeneous as well as heterogeneous tissue environments, we develop a permutation based localized weighted spatial autocorrelation metric which when applied to each spot efficiently selects the neighborhood within which to pool transcriptomic and spatial information to improve deconvolution predictions. We probabilistically align adjacent tissue slices in 3D and infer intermediate slice expression to augment our 3D spatial tissue slice stack. Utilizing a learnable parameter, we model the batch effect between the reference scRNA-seq and the spatial dataset. We define several metrics and perform downstream analyses to distill SpaDecoder cell type proportions and scRNA-seq-spatial maps into interpretable biological findings. Our preprint can be accessed at bioarxiv. An overview of SpaDecoder and downstream analyses is shown in the figure 1 of the manuscript.


- **Developed by:** Macrina Lobo
- **Model type:** SpaDecoder is an optimization-based non-deep learning method that uses a matrix factorization based objective function
- **Language(s):** Python
- **License:** GNU GENERAL PUBLIC LICENSE Version 3


### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/ZhangLabGT/spadecoder
- **Paper:** https://www.biorxiv.org/content/10.64898/2026.02.10.705204v1

## Uses

SpaDecoder is developed for deconvolution of multiple neighboring spot-based spatial transcriptomic slices. It utilizes a cell type annotated scRNAseq reference capturing a similar tissue region. For each spot, it outputs a scRNAseq reference cell by cell type soft assignment matrix, a spot correction scalar, and the deconvolution proportions for each spot. Its mathematical formulation enables its use a wide range of downstream biologically relevant tasks. From the cell type proportions for each spot, we define two metrics, a global (G-3DSCI) and local (L-3DSCI) 3D Cell Type Spatial Colocalization Index to reveal cell type colocalization patterns. We identify tissue regions with significantly varying cell type composition across slices with permutation testing and the Jenson-Shannon distance metric, and perform connected component analysis to capture distinct spatial regions corresponding to key cell types. We leverage SpaDecoder's reference scRNA-seq cell-to-spatial-spots mapping to annotate fine grained cell types in space, predict the 3D spatio-temporal location of scRNA-seq cells, denoise the expression of measured spatial genes, and predict the expression of unmeasured genes. SpaDecoder is designed for bioinformaticians and data analysts as well as method developers seeking ways to leverage 3D tissue structure for spatial tasks. 


## Bias, Risks, and Limitations

SpaDecoder may not work well if the spatial or temporal tissue slices are very far apart or if the reference scRNAseq dataset captures a very dissimilar tissue region. 


## How to Get Started with the Model

Run the file example/code/run_dataset_spadecoder.py to get started with the model. Alter the paths accordingly to run on your own data. 

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

SpaDecoder is an optimization-based non-deep learning method that uses a matrix factorization based objective function. Hence there is no separate training phase.
SpaDecoder was tested using 5 spatial transcriptomic datasets along with corresponding reference scRNAseq data. To overcome the lack of ground truth in spot-based spatial transcriptomic data, spots were simulated from single cell spatial transcriptomic data for quantitative evaluations. Details of datasets and simulations are included in Supplementary Tables 1 and 2. 

#### Preprocessing

##### Data Preprocessing
Genes present in less than 10 cells were excluded and the subset of overlapping genes between scRNA-seq and spatial data was selected. Cells with minimum total counts <50 were excluded. In case there were over 5000 resulting genes in the set, we performed additional gene feature selection. For scRNA-seq data, we normalize (sc.pp.normalize_total) and log pseudo count transform (sc.pp.log1p) the count matrix and select the differentially upregulated genes in each defined scRNA-seq cluster (adjusted p-value <0.05, log-foldchange >1.0). We concatenate all spatial slices and select the top 5000 slice-aware highly variable genes (sc.pp.highly_variable_genes) to avoid the selection of slice specific genes. The overlap between spatial highly variable genes and scRNA-seq differentially upregulated genes comprises the resulting feature set.
scRNA-seq and spatial data is normalized using sc.pp.normalize_total with target_sum =1000 to correct for varying sequencing depth. 
Since different spatial datasets might have different coordinates and spatial distances between spots, we scale X-Y spatial coordinates to [0,1].

##### SpaDecoder Preprocessing
SpaDecoder preprocessing comprises alignment of multiple slices with moscot, imputation of additional intermediate slices, per spot 3D neighborhood selection and spatio-transcriptomic kernel estimation.

#### Hyperparameters
The model is stable to a range of hyperparameters (Supplementary Fig. 9 in our manuscript). Hence we recommend the default hyperparameters. 

#### Metrics

SpaDecoder was evaluated with established metrics - average RMSE, JSD, and Pearson correlation. 

### Results

SpaDecoder outperformed benchmarked deconvolution methods (CARD, Cell2location, Tangram, single cell Tangram) 

## Model Examination

The reference scRNAseq cell to cell type association matrix and the deconvolution scores are highly interpretable and are used in several downstream biological applications as illustrated in the manuscript 

## Technical Specifications

### Model Architecture and Objective

SpaDecoder is an optimization-based non-deep learning method that uses a matrix factorization based objective function.

#### Hardware

SpaDecoder was run using a single A40 GPU wtih 46GB of memory. However, it is possible to run it on a CPU only. 

#### Software

SpaDecoder uses:  
1. python==3.12.3
2. scanpy==1.10.1 
3. squidpy==1.2.2
4. scipy==1.13.1
5. numpy==1.26.4
6. pandas==2.2.2
7. scikit-learn==1.5.0
8. moscot==0.4.2
9. matplotlib==3.9.0
10. seaborn==0.13.2
11. torch==2.3.1+cu121 - Make sure the torch version installed is compatible with your GPU

## Citation

Spatiotemporal cell type deconvolution leveraging tissue structure
Macrina Maria Lobo, Ziqi Zhang, Xiuwei Zhang
bioRxiv 2026.02.10.705204; doi: https://doi.org/10.64898/2026.02.10.705204

## Model Card Contact

Macrina Lobo (mlobo6 at gatech.edu)
