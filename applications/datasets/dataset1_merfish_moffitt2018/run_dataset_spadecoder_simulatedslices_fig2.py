#!/usr/bin/env python
# coding: utf-8
import sys
package_path = "/localscratch/mlobo6/spadecoder/datasets/"
if package_path not in sys.path:
    sys.path.append(package_path)
from spadecoder import *


result_metric = ['orig_rmse',   'avg_corr_pe','avg_jsd']

par_base = {###########. general path parameters ############################
            'adata_scrna_path':"../data/scrna_ref_normnolog_nov2024_1k.h5ad", # scRNA
            'simdir':'../results/simulations/slice_warps/', # where to write outputs 
            'suffix':'_normnolog_nov2024_1k',
            'spatial_distances_file':"../data/pairwise_slice_distances.csv",
            'result_ext':'_realalign_',
            'pickle_path':'../results/simulations/pickles/',
            #################################################################
            
            ########### simulation parameters  ##############################
            'N':50, # spot-size parameter
            'nswaps_nbd':2, # # number of cells swapped in a neighborhood  only for multi-slice sim
            'nneigh':10, # neighborhood for the cell swapping 
            #################################################################
            
            ##########  data parameters #####################################
            'scrna_cluster_key':"Cell class (determined from clustering of all cells)",
            'spa_key':'spatial',
            #################################################################
            
            ############## 2D kernel parameters ################################
            'par_lambda':0.1, #2 [0.0, 0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 100],
            'bandwidth':0.01, #[0.1] # tune  [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.0]
            'nn_only':True, # clipping to neighbors only 
            'n_spatial_neigh':10, #[5,10,15,20] # spatial neighbors within same slice [1, 3, 5, 10, 15, 20, 50, 75, 100]}
            'min_wt':0.0001,
            'renorm':True, # should I renormalize the kernel
            #################################################################

            ############## 3D kernel parameters ############################
            'kernel3d_bw_slices':8, # used onyl when inter slice distance insnt known so we set a fixed kernel
            'num_3D_neigh':11, # numbr of neighboring slices (excluding current )
            'bw_3D': 0.01, # 0.01, # 3D 
            ################ optimization  parameters ################################
            'max_iter_adam':500,
            'par_lr_adam':0.01,
            #################################################################

            ################## method #######################################
            'mode':'multislice', # singleslice
            'gt_align':False, # if True, align to single corresponding spot number of next slice, if False, run Alignment 
            'align_tool':'moscot', # try others 
            'key_name':'linear_0',
            'modeNbd':'variabletranscr',
            'spatial_neigh_max':30,
            'num_augment':20,
            'do_augment':True #True,
            # 'dist_between_slices':5
            }


par_to_iterate = {#'spatial_distances_file':"../data/pairwise_slice_distances.csv",
            #'kernel3d_bw_slices':100,
            # 'N':[ 10, 5], # 100,  75, 25, 
             'mode': [ 'multislice' ], # , #  # , ,singleslice 'multislice'
             'modeNbd':[ 'variabletranscr'],#[], #, #[], # 'variabletranscr' ,'variable', 'fixed']
              'align_tool':['moscot'],
                # 'n_spatial_neigh':[10, 20, 30]}
                'nswaps_nbd':[ 2, 10]}# , 20]}
              # 'num_augment':[1, 5, 10, 15, 20],
              # 'kernel3d_bw_slices':[2,4,8,16]}
              #'par_lambda':[0.0, 0.001, 0.01, 0.1, 1.0, 10.0],
              # 'bandwidth':[0.01, 0.1, 1.0, 10.0]}#, 'slat','fgw'] } # 100 # , , # 'moscot',

var_bw = [ 0.01, 0.1, 1.0, 2.5, 5.0, 10.0, 100.0] # [0.000001,  0.00001,  0.0001,  0.001, 0.01, 0.1,  5, 10, 100]
var_nspaneigh= [1, 2, 3, 4, 5, 10, 15, 20]
var_nspaneigh_transcr= list(range(1,par_base['spatial_neigh_max']))



def spadecoder_run( Bbulk,ct_identity_bulk,ct_props,adata_sc_df,ct_identity_sc, mode=par_base['mode'],**kwargs):
    
    # extract parameters, defaults otherwise 
    par_lambda = kwargs.get('par_lambda',par_base['par_lambda']) # 2
    bandwidth = kwargs.get('bandwidth', par_base['bandwidth'])
    n_spatial_neigh = kwargs.get('n_spatial_neigh', par_base['n_spatial_neigh'])
    spatial_distances_file = kwargs.get('spatial_distances_file', par_base['spatial_distances_file'])
    kernel3d_bw_slices = kwargs.get('kernel3d_bw_slices', par_base['kernel3d_bw_slices'])
    num_3D_neigh = kwargs.get('num_3D_neigh', par_base['num_3D_neigh'])
    bw_3D = kwargs.get('bw_3D', par_base['bw_3D'])
    N_curr = kwargs.get('N', par_base['N'])
    # mode = kwargs.get('mode', par_base['mode'])
    min_wt = kwargs.get('min_wt', par_base['min_wt'])
    nn_only = kwargs.get('nn_only', par_base['nn_only'])
    key_name = par_base['key_name']
    gt_align =  kwargs.get('gt_align', par_base['gt_align'])
    nneigh =  kwargs.get('nneigh', par_base['nneigh'])
    nswaps_nbd =  kwargs.get('nswaps_nbd', par_base['nswaps_nbd'])
    suffix =  kwargs.get('suffix', par_base['suffix'])
    align_tool =  kwargs.get('align_tool', par_base['align_tool'])
    result_ext = kwargs.get('result_ext', par_base['result_ext'])
    mode_nbd = kwargs.get('modeNbd', par_base['modeNbd'])
    num_augment = kwargs.get('num_augment', par_base['num_augment'])
    augment = kwargs.get('do_augment', par_base['do_augment'])

    partesting_str =     '_partesting_' + 'parlambda_' + str(par_lambda) + '_parbw_' + str(bandwidth) + '_nbdtype_' + mode_nbd + '_gtalign_' + str(gt_align) + '_augment_' + str(augment) + '_kernel3d_bw_slices_' + str(kernel3d_bw_slices) + '_num_augment_' + str(num_augment) # + '_pareta1_' + str(par_eta1) +  '_3Dkernelbw_' + str(kernel3d_bw_slices_curr)
    print(partesting_str)
    print(mode)
    print(align_tool)
    print(nswaps_nbd)


    #### read spots from semi-simulated spatial file ######
    adata_spa_path = par_base['pickle_path'] + 'multi_slice_simulated_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) + suffix + '.pickle' # '_old.pickle'        
    # read spatial file 
    if not os.path.exists(adata_spa_path):
        sys.exit() 
    with open(adata_spa_path, 'rb') as handle:
        adata_spa = pickle.load(handle)
    #################################################################


    slice_order = [int(entry) for entry in list(adata_spa[key_name].keys())]
    slice_order.sort()
    # slice_order = [str(entry) for entry in slice_order]
    # slice_list = [adata_spa[key_name][0][entry] for entry in slice_order] 
    
    ###### incorporate real spatial distances into z-axis kernel #########################
    # if (spatial_distances_file is not None) and (mode == 'multislice'):
    #     spatial_distances = pd.read_csv(spatial_distances_file,index_col = 0)
    #     adj_wts = get_gauss_kernel_3D(np.abs(spatial_distances.values), nneigh3D=num_3D_neigh, bandwidth=bw_3D,  min_wt = min_wt, nn_only=nn_only)
    # else:
    #     adj_wts = None
    #######################################################################################

    deconv_st1_bulk = {}
    deconv_st1_sc = {}
    results_st1_bulk = {} 
    results_st1_sc = {} 

    Palign_orig = {}
    Palign_dis = {}

    slice_list_extra = {}
    kernel_wt_extra = {}
    adj_wts_dict = {}

    kernel_all = {}
    geary_metric = {}

    for realslice in list(adata_spa[key_name][0].keys()):
        slice_list = [adata_spa[key_name][entry][realslice] for entry in slice_order] 
        
        results_st1_bulk[realslice] = pd.DataFrame(index=[metric_entry + '_avg' for metric_entry in result_metric])
        results_st1_sc[realslice] = pd.DataFrame(index=[metric_entry + '_avg' for metric_entry in result_metric])
        
        deconv_st1_bulk[realslice] = {}
        deconv_st1_sc[realslice] = {}

    # slice_order = slice_order[0:3]
    # slice_list = slice_list[0:3]

    ##################  alignment dictionary #########################
    
        if (not gt_align) and  (mode == 'multislice'): # need to align
            
            align_path =  par_base['pickle_path'] + align_tool + '_align_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  suffix + '_sim.pickle' # moscot_align_sptsz_50_nneigh_10_nbdswaps_2_normnolog_nov2024_1k.pickle' # moscot_me.pickle' #moscot_me.pickle' #par_base['pickle_path'] + align_tool + '_align_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  suffix + '.pickle'
            align_orig_path =  par_base['pickle_path'] + align_tool + '_alignorig_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  suffix + '_sim.pickle'
            
            # conn_path = par_base['pickle_path'] + 'slat' + '_align_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  suffix + '.pickle' # moscot_align_sptsz_50_nneigh_10_nbdswaps_2_normnolog_nov2024_1k.pickle' # moscot_me.pickle' #moscot_me.pickle' #par_base['pickle_path'] + align_tool + '_align_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  suffix + '.pickle'

            if os.path.exists(align_path): # align exists read 
                with open(align_path, 'rb') as handle:
                    Palign_dis = pickle.load(handle) # dictionary of alignments 
                with open(align_orig_path, 'rb') as handle:
                    Palign_orig = pickle.load(handle) # dictionary of alignments 
                # if align_tool == 'slat':
                #     with open(conn_path, 'rb') as handle:
                #         DSpaConn = pickle.load(handle) # dictionary of alignments 
            else: # generate align - dont run yet,
                if align_tool == 'moscot':
                    Palign_dis[realslice], Palign_orig[realslice] = get_align_moscot(slice_order, slice_list,min_wt=min_wt) # this is the distance NOT the prob
                    # DSpaConn = None
                if align_tool == 'slat':
                    Palign_dis[realslice] = get_align_slat(slice_order, slice_list) # this is the distance NOT the prob
                    # with open(conn_path, 'wb') as handle:
                    #     pickle.dump(DSpaConn, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if align_tool == 'fgw':
                    # single_slice_results = par_base['simdir'] + 'deconv_'  +  'singleslice' + '_sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  partesting_str + result_ext + '_' + align_tool +  '_sc_varbw.pickle'
                    single_slice_results = par_base['simdir'] + 'deconv_'  +  'singleslice' + '_' + mode_nbd + '_sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  partesting_str + result_ext + 'moscot' +  '_sc_sim.pickle'  # single slice has no align, so moscot is default 
                    with open(single_slice_results, 'rb') as handle:
                        adata_res_tmp = pickle.load(handle)
                    # generate the result list 
                    adata_results = [adata_res_tmp[realslice][entry] for entry in slice_order] 
                    Palign_dis[realslice] , Palign_orig[realslice]  = get_align_modifiedfgw(slice_order, slice_list, adata_results, min_wt=min_wt) # this is the distance NOT the prob
                    # DSpaConn = None
                
        else:# dont need alignment, set to identity matrix 
            Palign_dis[realslice] = {}
            # Palign_orig = None
            for slice1 in range(len(slice_list)):
                for slice2 in range(len(slice_list)):
                    if slice1 != slice2:
                        kernel_wt1 = np.zeros((slice_list[slice1].shape[0],slice_list[slice2].shape[0]))
                        np.fill_diagonal(kernel_wt1,1.0)
                        
                        Palign_dis[realslice][(str(slice1),str(slice2))] = kernel_wt1.copy()
            Palign_orig[realslice] = Palign_dis[realslice].copy() 
            
            
            
        ################################################################################################################


        # ################################################ 3D variable BW #############################################
        print("Getting kernel with " + mode_nbd)

        if mode_nbd == 'fixed':
            mode_nbd = mode_nbd + '_nspatialneigh_' + str(n_spatial_neigh) 
        start_time = time.time()
        kernel_path = par_base['pickle_path'] + '/kernel_wt_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd)  + '_parbw_' + str(bandwidth) + suffix + '_' + align_tool + '_' + mode + '_' + mode_nbd + '_gtalign_' + str(gt_align) + '_sim.pickle'
        geary_path = par_base['pickle_path'] + '/geary_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd)  + '_parbw_' + str(bandwidth) + suffix + '_' + align_tool + '_' + mode + '_' + mode_nbd + '_gtalign_' + str(gt_align) +  '_sim.pickle'
        
        if os.path.exists(kernel_path):
            with open(kernel_path,'rb') as handle:
                kernel_all = pickle.load(handle)
        else:
            if mode_nbd == 'variable':
            
                # construct it 
                kernel_all[realslice], geary_metric[realslice] = get_variable_bw(slice_order, slice_list,var_nspaneigh,var_bw, mode, Palign_dis=Palign_dis[realslice], min_wt=min_wt,method=align_tool, gt_align=gt_align)
                
                end_time = time.time() # took 2min 
                print(f"Runtime: {end_time - start_time:.3f} seconds")                
                
        
        ################################## transcriptomic variable kernel ###############################################################################   
            elif mode_nbd == 'variabletranscr':
            
                # construct it 
                kernel_all[realslice], geary_metric[realslice] = get_variabletranscr_bw(slice_order, slice_list,var_nspaneigh_transcr, mode, fixed_bw = bandwidth, Palign_dis=Palign_dis[realslice], min_wt=min_wt,method=align_tool, gt_align=gt_align)
                # print(kernel_all[realslice])
                end_time = time.time() # took 2min 
                print(f"Runtime: {end_time - start_time:.3f} seconds")                
                
        ################################## fixed kernel ###############################################################################      
            elif  'fixed' in mode_nbd: 
            
                # construct it 
                kernel_all[realslice] = get_fixed_bw(slice_order, slice_list, mode,  fixed_bw=bandwidth, fixed_nn=n_spatial_neigh, Palign_dis=Palign_dis[realslice], min_wt=min_wt,method=align_tool, gt_align=gt_align)
                
                end_time = time.time() # took 2min 
                print(f"Runtime: {end_time - start_time:.3f} seconds")                
                
        print("kerenl obtained")
        ###############################################################################################################################


        ###########################################################     DATA AUGMENTATION  ###################################################################
        if augment: 
            if mode == 'multislice':
                print("augmenting data")
                extra_slices= impute_slices(Palign_orig[realslice], slice_list, slice_order, num_mid=num_augment, chunk_size = 500, key_name=key_name)
                print("augmentation complete")
                # num_sim  = int(num_augment/2)
                slice_list_extra[realslice] = {}
                kernel_wt_extra[realslice] = {}
                adj_wts_dict[realslice] = {}
                # num_slices = adj_wts.shape[0]
                for tmp_slice_tgt in range(len(slice_order)):
                    # tgt_slice = slice_order[tmp_slice_tgt]
                    adj_wts_dict[realslice][tmp_slice_tgt] = [] # np.zeros(num_slices + num_augment* (num_slices-1))
                    # adj_wts_dict[tmp_slice_tgt][0:num_slices] = adj_wts[:,tmp_slice_tgt].copy()
                    slice_list_extra[realslice][tmp_slice_tgt] = []
                    kernel_wt_extra[realslice][tmp_slice_tgt] = []
                    # cnt_tmp = num_slices

                    # add previous slice
                    if tmp_slice_tgt != 0: # not the first slice 
                        
                        tmp_slice_src = tmp_slice_tgt - 1
                        # add previous slice 
                        slice_list_extra[realslice][tmp_slice_tgt].append(slice_list[tmp_slice_src])
                        kernel_wt_extra[realslice][tmp_slice_tgt].append(torch.from_numpy(kernel_all[realslice][ str(slice_order[tmp_slice_src]) + '_' + str(slice_order[tmp_slice_tgt])  ]).to(device))

                        # add simulated slices 
                        for tmp_sim in range(0,num_augment):
                            tmp_slice = slice_list[tmp_slice_tgt].copy() 
                            tmp_slice.X =  extra_slices[(str(slice_order[tmp_slice_src]), str(slice_order[tmp_slice_tgt]), str(tmp_sim))].copy()
                            slice_list_extra[realslice][tmp_slice_tgt].append(tmp_slice)
                            kernel_wt_extra[realslice][tmp_slice_tgt].append(torch.eye(kernel_all[realslice][ str(tmp_slice_tgt) + '_' + str(tmp_slice_tgt)  ].shape[0]).to(device)) #torch.from_numpy(kernel_all[ str(tgt_slice) + '_' + str(tgt_slice)  ]).to(device))
                    
                    # add current slice 
                    slice_list_extra[realslice][tmp_slice_tgt].append(slice_list[tmp_slice_tgt])
                    kernel_wt_extra[realslice][tmp_slice_tgt].append(torch.from_numpy(kernel_all[realslice][ str(tmp_slice_tgt) + '_' + str(tmp_slice_tgt)  ]).to(device))
            
                    # if its not the last, add a stack
                    if tmp_slice_tgt != (len(slice_list) - 1): # not the first slice 
                        
                        tmp_slice_src = tmp_slice_tgt + 1
                        
                        # add a bunch of simulated slices 
                        tmp_extra = []
                        for tmp_sim in range(0,num_augment):
                            tmp_slice = slice_list[tmp_slice_tgt].copy() 
                            tmp_slice.X =  extra_slices[(str(slice_order[tmp_slice_src]), str(slice_order[tmp_slice_tgt]), str(tmp_sim))].copy()
                            tmp_extra.append(tmp_slice)
                            kernel_wt_extra[realslice][tmp_slice_tgt].append(torch.eye(kernel_all[realslice][ str(tmp_slice_tgt) + '_' + str(tmp_slice_tgt)  ].shape[0]).to(device)) #torch.from_numpy(kernel_all[ str(tgt_slice) + '_' + str(tgt_slice)  ]).to(device))
                        # reverse so we start from "tgt"
                        tmp_extra.reverse()
                        slice_list_extra[realslice][tmp_slice_tgt].extend(tmp_extra)

                        # add the actual next slice
                        slice_list_extra[realslice][tmp_slice_tgt].append(slice_list[tmp_slice_src])
                        print( str(slice_order[tmp_slice_src]),str(slice_order[tmp_slice_tgt])  )# + '_' + str(slice_order[tmp_slice_tgt])  )
                        kernel_wt_extra[realslice][tmp_slice_tgt].append(torch.from_numpy(kernel_all[realslice][ str(slice_order[tmp_slice_src]) + '_' + str(slice_order[tmp_slice_tgt])  ]).to(device))

                    

                    ################################################################################# 3D Kernel  get weights ##################################################################
                    if tmp_slice_tgt == 0: # first slice 
                        slice_idx = 0
                    elif tmp_slice_tgt == (len(slice_list)-1): # last slice 
                        slice_idx = len(slice_list_extra[realslice][tmp_slice_tgt])-1
                    else:
                        slice_idx = int(len(slice_list_extra[realslice][tmp_slice_tgt])/2)
        
                    # slice_idx = int(len(slice_list_extra[tmp_slice_tgt])/2) # len(slice_list_extra[tmp_slice_tgt]) - 1
                    two_sig = int(kernel3d_bw_slices/2) # make sure the maximum number of important slices are covered in 2-sigma
                    sigma = two_sig/2.0
                    
                    for tmp_slice_idx in range(len(slice_list_extra[realslice][tmp_slice_tgt])):
                        dist_from_curr = abs(tmp_slice_idx-slice_idx)
                        slice_wt = gaussian_kernel_for3d(dist_from_curr, sigma)
                        #print(sigma, dist_from_curr, slice_idx,curr_slice_idx,slice_wt)
                        adj_wts_dict[realslice][tmp_slice_tgt].append(slice_wt)
                    adj_wts_dict[realslice][tmp_slice_tgt] = np.array(adj_wts_dict[realslice][tmp_slice_tgt])/np.sum(adj_wts_dict[realslice][tmp_slice_tgt]) # previously max

                    adj_wts_dict[realslice][tmp_slice_tgt] =  torch.tensor(adj_wts_dict[realslice][tmp_slice_tgt])
                    ################################################################ KERNEL WEIGHTS ##############################################################
            ######################################################################## DATA AUGMENTATION  ##############################################################


        else:
            print("no augmentation")
            slice_list_extra[realslice] = {}
            kernel_wt_extra[realslice] = {}
            adj_wts_dict[realslice] = {}      
            
            for tmp_slice_tgt in range(len(slice_order)):
                slice_list_extra[realslice][tmp_slice_tgt] = []
                adj_wts_dict[realslice][tmp_slice_tgt] = []
                kernel_wt_extra[realslice][tmp_slice_tgt] = []
                if tmp_slice_tgt != 0:
                    slice_list_extra[realslice][tmp_slice_tgt] = [slice_list[tmp_slice_tgt-1].copy()]
                    kernel_wt_extra[realslice][tmp_slice_tgt].append(torch.from_numpy(kernel_all[realslice][ str(tmp_slice_tgt-1) + '_' + str(tmp_slice_tgt)  ]).to(device))
                slice_list_extra[realslice][tmp_slice_tgt].append(slice_list[tmp_slice_tgt].copy())
                kernel_wt_extra[realslice][tmp_slice_tgt].append(torch.from_numpy(kernel_all[realslice][ str(tmp_slice_tgt) + '_' + str(tmp_slice_tgt)  ]).to(device))
                if tmp_slice_tgt != (len(slice_list)-1):
                    slice_list_extra[realslice][tmp_slice_tgt].append(slice_list[tmp_slice_tgt+1].copy())
                    kernel_wt_extra[realslice][tmp_slice_tgt].append(torch.from_numpy(kernel_all[realslice][ str(tmp_slice_tgt+1) + '_' + str(tmp_slice_tgt)  ]).to(device))

                # get weights 
                if tmp_slice_tgt == 0: # first slice 
                        slice_idx = 0
                elif tmp_slice_tgt == (len(slice_list)-1): # last slice 
                        slice_idx = len(slice_list_extra[realslice][tmp_slice_tgt])-1
                else:
                        slice_idx = int(len(slice_list_extra[realslice][tmp_slice_tgt])/2)
        
                # slice_idx = int(len(slice_list_extra[tmp_slice_tgt])/2) # len(slice_list_extra[tmp_slice_tgt]) - 1
                two_sig = int(kernel3d_bw_slices/2) # make sure the maximum number of important slices are covered in 2-sigma
                sigma = two_sig/2.0
                
                for tmp_slice_idx in range(len(slice_list_extra[realslice][tmp_slice_tgt])):
                    dist_from_curr = abs(tmp_slice_idx-slice_idx)
                    slice_wt = gaussian_kernel_for3d(dist_from_curr, sigma)
                    #print(sigma, dist_from_curr, slice_idx,curr_slice_idx,slice_wt)
                    adj_wts_dict[realslice][tmp_slice_tgt].append(slice_wt)
                adj_wts_dict[realslice][tmp_slice_tgt] = np.array(adj_wts_dict[realslice][tmp_slice_tgt])/np.sum(adj_wts_dict[realslice][tmp_slice_tgt]) # previously max

                adj_wts_dict[realslice][tmp_slice_tgt] =  torch.tensor(adj_wts_dict[realslice][tmp_slice_tgt])

        print("HERE")
        

        for currslice_idx in range(len(slice_order)): # iterate over real input samples or slices  
            currslice_idx_name = slice_order[currslice_idx]

            # results_st1_bulk[currslice_idx_name] = pd.DataFrame(index=[metric_entry + '_avg' for metric_entry in result_metric])
            # results_st1_sc[currslice_idx_name] = pd.DataFrame(index=[metric_entry + '_avg' for metric_entry in result_metric])
            
            # deconv_st1_bulk[currslice_idx_name] = {}
            # deconv_st1_sc[currslice_idx_name] = {}

            if mode != 'multislice':
                slice_list_curr = [adata_spa[key_name][currslice_idx_name][realslice]]
                slice_idx = 0 
                adj_wts_curr = None
                kernel_wt_list = [torch.from_numpy(kernel_all[realslice][ str(slice_order[currslice_idx]) + '_' + str(slice_order[currslice_idx])  ]).to(device)]
            else:
                slice_list_curr = slice_list_extra[realslice][currslice_idx]
                kernel_wt_list = kernel_wt_extra[realslice][currslice_idx]
                # slice_idx = int(len(slice_list_curr)/2) #len(slice_list_curr) - 1

                if currslice_idx == 0: # first slice 
                    slice_idx = 0
                elif currslice_idx == (len(slice_list)-1): # last slice 
                    slice_idx = len(slice_list_curr)-1
                else:
                    slice_idx = int(len(slice_list_curr)/2)
    
                adj_wts_curr = adj_wts_dict[realslice][currslice_idx]

    
            # print(adj_wts_curr)
            # print(currslice_idx, slice_idx, len(slice_list_curr))

            print("bulk " + "real slice " + str(realslice) + "sim slice " + str(currslice_idx_name))

            # single_slice_results = '../results/simulations/slice_warps/' + 'deconv_' + 'singleslice' + '_sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  '_partesting_parlambda_0.1_parbw_0.01_PASTEreal_kernel_realdist_cluster.pickle'

            # with open(single_slice_results, 'rb') as handle:
            #     adata_ss = pickle.load(handle)
            
            
            deconv_st1_bulk[realslice][currslice_idx_name], _, _   = spadecoder_slice_wrapper_ms(slice_list_curr, slice_idx,
                                                        # int(len(slice_list)/2), # index of slice we're predicting for
                                                        Bbulk,ct_identity_bulk, 
                                                        min_wt=min_wt,
                                                        # spa_key1=spa_key1,
                                                        renorm=par_base['renorm'],# recompute=recompute,
                                                        bandwidth=bandwidth,
                                                        par_lambda=par_lambda,
                                                        max_iter_adam=par_base['max_iter_adam'],
                                                        n_spatial_neigh=n_spatial_neigh,nn_only=nn_only,
                                                        par_lr_adam=par_base['par_lr_adam'],
                                                        # weight_spatial=weight_spatial,
                                                        ct_props= ct_props, 
                                                        # adata_bulk_init=adata_ss[currslice_idx_name][0],
                                                        # par_eta1=par_eta1, # keep_top=5,
                                                        kernel3d_bw_slices=kernel3d_bw_slices,# Palign_ip = Palign_dis,
                                                        gt_align=gt_align, adj_wts =  adj_wts_curr,
                                                        kernel_wt_list = kernel_wt_list,
                                                        cell_mode = 'cluster')      
                                                    #   kernel_ip = kernel_wt[key_name][entry3][currslice_idx])
            
            results_st1_bulk[realslice][currslice_idx_name], _ = eval_deconv3(adata_spa[key_name][currslice_idx_name][realslice].obs,deconv_st1_bulk[realslice][currslice_idx_name] )
            
            print(results_st1_bulk[realslice][currslice_idx_name])

            print("sc " + "real slice " + str(realslice) + "sim slice " + str(currslice_idx_name))

            
            deconv_st1_sc[realslice][currslice_idx_name], _, _ = spadecoder_slice_wrapper_ms(slice_list_curr, slice_idx,
                                                    # int(len(slice_list)/2),
                                                    adata_sc_df,ct_identity_sc, #Palign='no neigh', 
                                                    min_wt=min_wt,
                                                    # spa_key1=spa_key1,
                                                    renorm=par_base['renorm'],# recompute=recompute,
                                                    bandwidth=bandwidth,
                                                    par_lambda=par_lambda,
                                                    max_iter_adam=par_base['max_iter_adam'],
                                                    n_spatial_neigh=n_spatial_neigh,nn_only=nn_only,
                                                    par_lr_adam=par_base['par_lr_adam'],
                                                    # weight_spatial=weight_spatial,
                                                     ct_props=ct_props,
                                                    # ct_props=None,
                                                    # par_eta1=par_eta1, # ap=ap_curr, # keep_top=5,
                                                    adata_bulk_init=deconv_st1_bulk[realslice][currslice_idx_name],
                                                    kernel3d_bw_slices=kernel3d_bw_slices,# Palign_ip = Palign_dis,
                                                        gt_align=gt_align, adj_wts = adj_wts_curr,
                                                        kernel_wt_list = kernel_wt_list,
                                                        cell_mode = 'sc') 
                                                        # kernel_ip = kernel_wt[key_name][entry3][currslice_idx])             

            results_st1_sc[realslice][currslice_idx_name], _ = eval_deconv3(adata_spa[key_name][currslice_idx_name][realslice].obs,deconv_st1_sc[realslice][currslice_idx_name])
            
            print(results_st1_sc[realslice][currslice_idx_name])


        

    write_slice1 = par_base['simdir'] + 'deconv_'  +  mode + '_' + mode_nbd + '_sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  partesting_str + result_ext +  align_tool +  '_sc_sim.pickle'
    with open(write_slice1, 'wb') as handle:
            pickle.dump(deconv_st1_sc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    write_slice1 = par_base['simdir'] + 'deconv_'  + mode + '_' + mode_nbd + '_sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  partesting_str + result_ext +  align_tool + '_cluster_sim.pickle'
    with open(write_slice1, 'wb') as handle:
            pickle.dump(deconv_st1_bulk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metrics_slice1  = par_base['simdir'] + 'metrics_' + mode + '_' + mode_nbd + '_sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  partesting_str +  result_ext + align_tool + '_sc_sim.pickle'
    with open(metrics_slice1, 'wb') as handle:
        pickle.dump(results_st1_sc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metrics_slice1  = par_base['simdir'] + 'metrics_' + mode + '_' + mode_nbd + '_sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd) +  partesting_str + result_ext +  align_tool + '_cluster_sim.pickle'
    with open(metrics_slice1, 'wb') as handle:
        pickle.dump(results_st1_bulk, handle, protocol=pickle.HIGHEST_PROTOCOL)


    if ((mode == 'multislice') and (not gt_align)): # if we actually aligned it and we're in multislice mode
        if not os.path.exists(align_path):
            with open(align_path, 'wb') as handle:
                pickle.dump(Palign_dis, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if Palign_orig is not None: # only for fgw, moscot
            with open(align_orig_path, 'wb') as handle:
                pickle.dump(Palign_orig, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not os.path.exists(kernel_path):
        with open(kernel_path,'wb') as handle:
            pickle.dump(kernel_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if 'fixed' not in mode_nbd: # no geary for fixed 
            with open(geary_path,'wb') as handle:
                pickle.dump(geary_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)

    adj_wt_path = par_base['pickle_path'] + '/adj_wts_'  + mode + '_' + mode_nbd + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd)  +  partesting_str + result_ext +  align_tool + '_adjwts_sim.pickle'
    with open(adj_wt_path,'wb') as handle:
        pickle.dump(adj_wts_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ############################       if fgw run for a couple of more iterations  ######################################### 
    # if ((not gt_align) and  (mode == 'multislice') and (align_tool == 'fgw')):
    #     # run for a couple of more iterations 
        
    ########################################################################################################################    
    return 

if __name__ == "__main__":
    
    # read scRNA file 
    if not os.path.exists(par_base['adata_scrna_path']):
        sys.exit() 

    adata_scrna = sc.read(par_base['adata_scrna_path'])

    Bbulk, ct_identity_sc = get_ref_for_spatialloc(adata_scrna, par_base['scrna_cluster_key'])
    # get ref cell-type proportions
    ct_props = get_ct_props_in_ref(adata_scrna,Bbulk,ct_key=par_base['scrna_cluster_key'])
    ct_identity_bulk = pd.DataFrame(np.diag(ct_props), index=list(ct_props.index), columns=list(ct_props.index))
    ct_identity_bulk = ct_identity_bulk.loc[Bbulk.columns, Bbulk.columns]

    # for mode_curr in par_dict['mode']: # singleslice, multislice 
    #     for par, par_values in par_dict.items():
    #         if mode_curr == 'singleslice' and par == 'align_tool': # no need to perform alignment in single slice mode
    #             par_values = ['moscot']
    #         if par != 'mode': # this must be run on all 
    #             for par_value in par_values:
                
    #                 spadecoder_run(  Bbulk,ct_identity_bulk,ct_props,adata_scrna.to_df().T,ct_identity_sc,mode=mode_curr,**{par:par_value}) # ,par_lambda=par_lambda)

            
    for mode_curr in par_to_iterate['mode']: # singleslice, multislice
        for modeNbd_curr in par_to_iterate['modeNbd']: # fixed, variable, variabletranscr
            if mode_curr == 'multislice':
                for align_tool_curr in par_to_iterate['align_tool']:
                    # for par_tune in par_to_iterate.keys():
                    #     if par_tune not in ['mode','modeNbd','align_tool']:
                    #         for par_value in par_to_iterate[par_tune]:
                    # for num_augment in par_to_iterate['num_augment']:
                    #     for kernel3d_bw_slices in par_to_iterate['kernel3d_bw_slices']:
                    # spadecoder_run(  Bbulk,ct_identity_bulk,ct_props,adata_scrna.to_df().T,ct_identity_sc,mode=mode_curr,**{'modeNbd':modeNbd_curr, 'align_tool':align_tool_curr, 'num_augment':num_augment, 'kernel3d_bw_slices':kernel3d_bw_slices})# par_tune:par_value}) 
                    for nswaps_nbd_curr in par_to_iterate['nswaps_nbd']:
                        spadecoder_run(  Bbulk,ct_identity_bulk,ct_props,adata_scrna.to_df().T,ct_identity_sc,mode=mode_curr,**{'modeNbd':modeNbd_curr, 'align_tool':align_tool_curr, 'nswaps_nbd':nswaps_nbd_curr})# par_tune:par_value}) 
            else:
                # for par_tune in par_to_iterate.keys():
                #         if par_tune not in ['mode','modeNbd','align_tool']:
                #             for par_value in par_to_iterate[par_tune]:
                # for num_augment in par_to_iterate['num_augment']:
                #     for kernel3d_bw_slices in par_to_iterate['kernel3d_bw_slices']:
                #        spadecoder_run(  Bbulk,ct_identity_bulk,ct_props,adata_scrna.to_df().T,ct_identity_sc,mode=mode_curr,**{'modeNbd':modeNbd_curr, 'num_augment':num_augment, 'kernel3d_bw_slices':kernel3d_bw_slices })# par_tune:par_value}) 
                for nswaps_nbd_curr in par_to_iterate['nswaps_nbd']:
                    spadecoder_run(  Bbulk,ct_identity_bulk,ct_props,adata_scrna.to_df().T,ct_identity_sc,mode=mode_curr,**{'modeNbd':modeNbd_curr, 'align_tool':'moscot', 'nswaps_nbd':nswaps_nbd_curr})# par_tune:par_value}) 
            
