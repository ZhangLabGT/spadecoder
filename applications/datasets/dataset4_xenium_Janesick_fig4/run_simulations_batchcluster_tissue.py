
########### simulation parameters  ##############################

adata_st_path = '../data/spatial_raw.pickle'# '..//data/spatial_data_normcorrect.pickle' # '..//data/spatial_data.pickle' # '..//data/spatial_data_normcorrect.pickle' - previously used this, changed on nov 18, 2024
N_range = [50] # [5, 10, 50,  100,  200,  400] #[5, 50, 500] # [5, 10, 20] # , 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500] # [500] # [5, 10, 50] # [100, 250] # [500]   # [5, 10, 20, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
nswaps_nbd_range = [2]#, 5, 10, 20, 50, 100] # [1, 2, 16, 50] # , 16, 50] # , 16] #[0, 2, 8, 16, 64, 250]
nneigh_range = [10]
wts = None 
spatial_base_key = 'spatial'
num_tissueslices = 10
spot_base_key = 'spot_number'
spa_celltype_key='Anno'


### package / libraries 
import sys
package_path = "/localscratch/mlobo6/spadecoder/datasets"
if package_path not in sys.path:
    sys.path.append(package_path)
from spadecoder.code import *
### to run as many processes 
from multiprocessing import Process
from multiprocessing import Semaphore

##### Main code 
def run_simulations(N,  savesimdir, adata_st,  nswaps_nbd = 5, n_neigh = 10):
    
    # some other variables 
    save_sim_file = savesimdir + 'multi_slice_simulated_' + 'sptsz_' + str(N)  + '_nneigh_' + str(n_neigh) + '_nbdswaps_' + str(nswaps_nbd) + '.pickle' # '_old.pickle'
    
    pct_cells_swapped, pct_spots_changed = sim_multiple_slices_allsims_tissue_forbatch_nowarp(adata_st,
                                            spa_celltype_key=spa_celltype_key,
                                            N=N, 
                                            save_sim_slices=save_sim_file,
                                            wts=wts,
                                            spatial_base_key = spatial_base_key,
                                            num_tissueslices = num_tissueslices,
                                            spot_base_key=spot_base_key,
                                            nswaps_nbd = nswaps_nbd,
                                            n_neigh = n_neigh)
    
    # pct_spots_changed['sptsz_' + str(N)  + '_nneigh_' + str(n_neigh) + '_nbdswaps_' + str(nswaps_nbd)] = pct_spots_changed_tmp
    # pct_cells_swapped['sptsz_' + str(N)  + '_nneigh_' + str(n_neigh) + '_nbdswaps_' + str(nswaps_nbd)] = pct_cells_swapped_tmp

    sema.release()  
    return  # pct_cells_swapped, pct_spots_changed


if __name__ == "__main__":
    savesimdir = '../results/simulations/pickles/' # '../results/simulations/pickles2/'
    os.makedirs(savesimdir, exist_ok=True)

    args = []
    procs = []

    with open(adata_st_path, 'rb') as handle:
        adata_st = pickle.load(handle)

    pct_cells_swapped = {}
    pct_spots_changed = {}

    for N in N_range:
        for nswaps_nbd in nswaps_nbd_range: # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for n_neigh in nneigh_range:#[0, 2, 4, 8, 16, 32, 64, 128, 256]:
                # pct_cells_swapped[str(N)+'_' + str(nswaps_nbd) + '_' + str(n_neigh)], pct_spots_changed[str(N)+'_' + str(nswaps_nbd) + '_' + str(n_neigh)]  = run_simulations(N, savesimdir,adata_st,  nswaps_nbd, n_neigh)
                 args.append((N, savesimdir,adata_st,  nswaps_nbd, n_neigh))

    concurrency = 30
    sema = Semaphore(concurrency)
    for arg in args:
        # print(name)
        sema.acquire()
        proc = Process(target=run_simulations, args=arg)
        procs.append(proc)
        proc.start()
   
   
    for proc in procs:
        proc.join()
        


    