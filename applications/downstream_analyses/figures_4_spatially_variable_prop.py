import sys
package_path = "/localscratch/mlobo6/spadecoder/datasets/"
if package_path not in sys.path:
    sys.path.append(package_path)
from spadecoder import *

## visualize spots + pie plots of results 


plt.rcParams['figure.figsize']=(8,8) #rescale figures
plt.rcParams['pdf.fonttype'] = 42


figdir = '../fig4a_main_realslices/'
sc.settings.figdir = '../fig4a_main_realslices/'


dataset = '12'

scrna_cluster_key = "Anno"

# deconv_dir = figdir + dataset + '/deconv/'
# metrics_dir = figdir + dataset + '/metrics/'

dataset = 'Janesick2023'
dataset_full = 'dataset12_xenium'
pickle_path = '../../datasets/' + dataset_full + '/results/simulations/pickles/'

N = 50
nneigh = 10
nbdswaps = 2
par_lambda = 0.1
par_bw = 0.01
augment = True
gt_align = False
kernel3d_bw_slices = 8
bandwidth = 0.01 
num_augment = 20
suffix = '_norm1knolog'



celltypes = ['BCells', 'DCs', 'Endothelial', 'Macrophages', 'Mast', 'Myoepithelial',
       'Perivascular-Like', 'Stromal', 'TCells', 'Tumor']

seed = 0 
nsim =  1000
batch_sz = 5


sig_thresh = 0.05 


################################    VARIABLES END ################################################################

deconv_dir = figdir  + '/deconv/'
# adata_sc = sc.read('../../datasets/' + dataset_full +   "/data/scrna_ref" + suffix + ".h5ad")

# cell_type = scrna_cluster_key
# celltype_colors = list(adata_sc.uns[cell_type + '_colors'])
# celltype_classes  = list(adata_sc.obs[cell_type].cat.categories)


adata_spa_path = figdir  + '/multi_slice_simulated_' + 'sptsz_' + str(N)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nbdswaps) + suffix + '.pickle' # '_old.pickle'

key_name = 'linear_0'

result_metric = ['orig_rmse',   'avg_corr_pe','avg_jsd']

# ground truth 
with open(adata_spa_path, 'rb') as handle:
    adata_spa = pickle.load(handle)



mode = 'multislice'
mode_nbd = 'variabletranscr'
aligntool = 'moscot'
# partesting = '_partesting_parlambda_' + str(par_lambda) + '_parbw_' + str(par_bw) + '_' + 'nbdtype_' + mode_nbd + '_realalign_' + aligntool
partesting =  '_partesting_' + 'parlambda_' + str(par_lambda) + '_parbw_' + str(bandwidth) + '_nbdtype_' + mode_nbd + '_gtalign_' + str(gt_align) + '_augment_' + str(augment) + '_kernel3d_bw_slices_' + str(kernel3d_bw_slices) + '_num_augment_' + str(num_augment) + '_realalign_' + aligntool # + '_pareta1_' + str(par_eta1) +  '_3Dkernelbw_' + str(kernel3d_bw_slices_curr)
res_file = 'deconv_' + mode + '_' + mode_nbd + '_sptsz_' + str(N) + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nbdswaps) + partesting + '_sc.pickle' 
with open(figdir + '/deconv/' + res_file, 'rb') as handle:
    deconv_ms = pickle.load(handle)

all_algos = {}

all_algos['SpaDecoder'] = deconv_ms

################################    READ  FILES ################################################################



################################    ADD 3D COORINDATES ################################################################

spadict = {} 
for slice in all_algos['SpaDecoder'].keys():
    # dim3 = adata_st[slice].obsm['spatial3d'][0,2]
    curr_spa = adata_spa['linear_0'][0][slice].obsm['spatial']
    new_column = np.full((curr_spa.shape[0],), slice*5)
    result_array = np.column_stack((curr_spa, new_column))
    adata_spa['linear_0'][0][slice].obsm['spatial3d'] = result_array

    spadict[str(slice)] = adata_spa['linear_0'][0][slice].copy()
    
    tmp_df = all_algos['SpaDecoder'][slice][0].T.loc[spadict[str(slice)].obs.index,:]

    spadict[str(slice)].obs = tmp_df.copy()

    spadict[str(slice)].obs['batch'] = str(slice)

slice_start = {}
curr_slice_start = 0
for slice in range(2):
    slice_start[str(slice)] = curr_slice_start
    curr_slice_start = curr_slice_start + spadict[str(slice)].shape[0]
slice_start['2'] =     curr_slice_start

slice_stacks = {} 
slice_stacks[str(0)] = anndata.concat([spadict['0'],spadict['1']])
slice_stacks[str(0)].obs['batch'] = slice_stacks[str(0)].obs['batch'].astype('category')
slice_stacks[str(1)] = slice_stacks[str(0)].copy()
slice_indices = {}
slice_indices['0'] = [0, slice_start['2']]
slice_indices['1'] = [0, slice_start['2']]
################################    ADD 3D COORINDATES ENDS ################################################################


################################    ALIGN SLICES, COMPUTE NEIGHBORS AND MORANS METRICS  ################################################################

# global morans 
# sig_celltypes = {} 
# moransI = {} 

# # local morans 
# local_moransI = {}
# signed_local_moransI = {}
# is_sig_local = {}
# pval_local = {}

# # global sci 
# sci = {}
# sci_pvals = {}
# sci_sig_celltypes = {}

# # local sci
# local_sci = {}
# local_sci_sig_celltypes = {}
# local_sci_pvals = {}

global_neigh_conn = np.zeros((slice_start['2'], slice_start['2']))

for entry in slice_stacks:
    ap = AlignmentProblem(adata=slice_stacks[entry])
    ap = ap.prepare(batch_key="batch", policy="sequential")
    ap = ap.solve()
    ap.align(reference=str(entry), key_added="seq_warp")

    slice_stacks[entry].obsm['seq_warp_3d'] = np.column_stack((slice_stacks[entry].obsm['seq_warp'],slice_stacks[entry].obsm['spatial3d'][:,2]))

    # sq.pl.spatial_scatter(
    # slice_stacks[entry],
    # shape=None,
    # spatial_key="seq_warp",
    # library_id="batch",
    # color="batch",
    # title="Alignment " + ':' + entry,# save='_samples.pdf',
    #   frameon=False
    # )

    sq.gr.spatial_neighbors(slice_stacks[entry], coord_type="generic", spatial_key="seq_warp_3d",delaunay=True) # spatial neighbors is by column here 

    spa_conn_curr = slice_stacks[entry].obsp['spatial_connectivities'].toarray()
    start_index = list(slice_stacks[entry].obs['batch'].values).index(entry)
    if str(int(entry)+1) in list(slice_stacks[entry].obs['batch'].values):
        end_index = list(slice_stacks[entry].obs['batch'].values).index(str(int(entry)+1))
    else:
        end_index = slice_stacks[entry].shape[0]
    global_neigh_conn[slice_indices[entry][0]:slice_indices[entry][1],slice_start[entry]:slice_start[str(int(entry)+1)]] = spa_conn_curr[:, start_index:end_index]


for entry in slice_stacks:
    slice_stacks[entry] = slice_stacks[entry][slice_stacks[entry].obs['batch'] == entry]

concatenated_adata = anndata.concat(slice_stacks)

concatenated_adata.obsp['spatial_connectivities'] = global_neigh_conn

##################################      GLOBAL MORANS I (AUTOCORRELATION) ########################################################
moransI, moransI_pvalues, moransI_pvalues_both, sig_celltypes, sig_celltypes_both = morans_I_permutation(concatenated_adata,celltypes,  seed=seed, nsim=nsim, batch_sz=batch_sz, sig_thresh=sig_thresh )

##################################      LOCAL MORANS I  (AUTOCORRELATION) ########################################################
local_moransI, signed_local_moransI, is_sig_local, is_sig_local_both, pval_local, pval_local_both = local_morans_I_permutation(concatenated_adata,celltypes,  seed=seed, nsim=nsim, batch_sz=batch_sz, sig_thresh=sig_thresh )

##################################      GLOBAL MORANS (SCI) CROSS CORRELATION   ########################################################
# batch_sz = 1
sci, sci_pvals, sci_pvals_both, sci_sig_celltypes, sci_sig_celltypes_both = sci_permutation(concatenated_adata,celltypes,  seed=seed, nsim=nsim, batch_sz=batch_sz, sig_thresh=sig_thresh )
##################################      LOCAL MORANS (SCI) CROSS CORRELATION   ########################################################
batch_sz = 1
local_sci, local_sci_sig_celltypes, local_sci_sig_celltypes_both, local_sci_pvals, local_sci_pvals_both = local_sci_permutation(concatenated_adata, celltypes,  seed=seed, nsim=nsim, batch_sz=batch_sz, sig_thresh=sig_thresh)
################################    ALIGN SLICES, COMPUTE NEIGHBORS AND MORANS METRICS  ENDS ################################################################

################################    WRITING OUTPUTS  ################################################################

write_slice1 = figdir  + 'global_morans_I.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(moransI, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
write_slice1 = figdir  + 'global_morans_I_pvalues.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(moransI_pvalues, handle, protocol=pickle.HIGHEST_PROTOCOL)

write_slice1 = figdir  + 'global_morans_I_pvalues_both.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(moransI_pvalues_both, handle, protocol=pickle.HIGHEST_PROTOCOL)

write_slice1 = figdir  + 'global_morans_I_sigcelltypes.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(sig_celltypes, handle, protocol=pickle.HIGHEST_PROTOCOL)

write_slice1 = figdir  + 'global_morans_I_sigcelltypes_both.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(sig_celltypes_both, handle, protocol=pickle.HIGHEST_PROTOCOL)

# local
write_slice1 = figdir  + 'local_morans_I.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(local_moransI, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

write_slice1 = figdir  + 'signed_local_morans_I.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(signed_local_moransI, handle, protocol=pickle.HIGHEST_PROTOCOL)

write_slice1 = figdir  + 'is_sig_local_morans_I.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(is_sig_local, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
write_slice1 = figdir  + 'is_sig_local_morans_I_both.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(is_sig_local_both, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

write_slice1 = figdir  + 'pvalue_local_morans_I.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(pval_local, handle, protocol=pickle.HIGHEST_PROTOCOL)

write_slice1 = figdir  + 'pvalue_local_morans_I_both.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(pval_local_both, handle, protocol=pickle.HIGHEST_PROTOCOL)


# global sci 
write_slice1 = figdir  + 'global_sci.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(sci, handle, protocol=pickle.HIGHEST_PROTOCOL)

write_slice1 = figdir  + 'global_sci_pvals.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(sci_pvals, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
write_slice1 = figdir  + 'global_sci_pvals_both.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(sci_pvals_both, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

write_slice1 = figdir  + 'global_sci_sig_celltypes.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(sci_sig_celltypes, handle, protocol=pickle.HIGHEST_PROTOCOL)

write_slice1 = figdir  + 'global_sci_sig_celltypes_both.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(sci_sig_celltypes_both, handle, protocol=pickle.HIGHEST_PROTOCOL)



# local sci
write_slice1 = figdir  + 'local_sci.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(local_sci, handle, protocol=pickle.HIGHEST_PROTOCOL)

write_slice1 = figdir  + 'local_sci_pvals.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(local_sci_pvals, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
write_slice1 = figdir  + 'local_sci_pvals_both.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(local_sci_pvals_both, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

write_slice1 = figdir  + 'local_sci_sig_celltypes.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(local_sci_sig_celltypes, handle, protocol=pickle.HIGHEST_PROTOCOL)

write_slice1 = figdir  + 'local_sci_sig_celltypes_both.pickle'
with open(write_slice1, 'wb') as handle:
    pickle.dump(local_sci_sig_celltypes_both, handle, protocol=pickle.HIGHEST_PROTOCOL)
################################    WRITING OUTPUTS ENDS  ################################################################
