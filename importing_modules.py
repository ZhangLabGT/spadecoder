import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


import math
import numpy as np
import scipy as sp
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

import moscot as mt
from moscot import datasets
from moscot.problems.space import MappingProblem
from moscot.problems.space import AlignmentProblem
from moscot.problems.generic import FGWProblem

import seaborn as sns
import pickle

# import time
# timestr = time.strftime("%Y%m%d-%H%M%S") # for timestamping output files 
# # print timestr



warnings.simplefilter("ignore", UserWarning)



import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib import colors
# plt.rcParams['figure.figsize']=(8,8) #rescale figures
# # sc.settings.verbosity = 3
# # sc.logging.print_versions()
# sc.set_figure_params(scanpy=True, dpi_save=400,dpi=150)
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams['pdf.fonttype'] = 42

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import time 

# torch.use_deterministic_algorithms(True)


import scSLAT
from scSLAT.model import run_SLAT_multi
from scSLAT.viz import build_3D
from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match
from scSLAT.viz import match_3D_multi, hist, Sankey

from sparsemax import Sparsemax