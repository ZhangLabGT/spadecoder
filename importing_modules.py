import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


import math
import numpy as np
import scipy as sp
import anndata
from scipy.stats import multivariate_normal as mvnpy
import random 

import pandas as pd

import anndata 

import scanpy as sc 

import squidpy as sq

from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from scipy.optimize import minimize
from scipy.stats import entropy


## perform OT mapping 
import os
# import warnings

# import moscot as mt
from moscot import datasets
from moscot.problems.space import MappingProblem
from moscot.problems.space import AlignmentProblem

import seaborn as sns
import pickle

# import time
# timestr = time.strftime("%Y%m%d-%H%M%S") # for timestamping output files 
# # print timestr



warnings.simplefilter("ignore", UserWarning)



import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
# plt.rcParams['figure.figsize']=(8,8) #rescale figures
# # sc.settings.verbosity = 3
# # sc.logging.print_versions()
# sc.set_figure_params(scanpy=True, dpi_save=400,dpi=150)
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams['pdf.fonttype'] = 42

import torch
import torch.nn.functional as F

torch.use_deterministic_algorithms(True)


