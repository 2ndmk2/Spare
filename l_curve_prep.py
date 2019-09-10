import healpy as hp
import pylab 
import matplotlib.pyplot as plt
import time
from tsv_healpix import * 
from tikhonov_solve import *
from geometry import *
import datetime
import logging
import log_make
import parser
import argparse



#python l_curve_prep.py -root_f "./result_l_curve/" -f "data_for_lcurve" -l1 -3 -3 -ltsv -5 0 -step_ltsv 0.25
parser = argparse.ArgumentParser()

parser.add_argument("-root_f", default='./result/', help="root folder")
parser.add_argument("-f", default='cv', help="Name of folder including result")
parser.add_argument("-l1", nargs = 2, default=[-4, 1], help="[l1min, l1max]")
parser.add_argument("-ltsv", nargs = 2, default=[-4, 1], help="[ltsvmin, ltsvmax]")
parser.add_argument("-step_l1",  type=float,default=1, help="step_l1")
parser.add_argument("-step_ltsv",type=float, default=1, help="step_ltsv")
parser.add_argument("-cv", action='store_true')
parser.add_argument("-debug", action='store_true')
parser.add_argument("-log_debug", action='store_true')


args = parser.parse_args()
l1_parser_min, l1_parser_max = tuple(args.l1)
ltsv_parser_min, ltsv_parser_max = tuple(args.ltsv)
one_step_for_l1 = float(args.step_l1)
one_step_for_ltsv = float(args.step_ltsv)
folder_branch = str(args.f)
folder_root_result = str(args.root_f)

## Creating output folders
make_folders(folder_root_result, folder_branch)


## Constructing logger
log_folder =folder_root_result + folder_branch + "/log/"

if not args.log_debug:
    logger = log_make.log_make("main",log_folder, logging.INFO)
else:
    logger = log_make.log_make("main",log_folder, logging.DEBUG)


# Loading data (Npix = 12 * nside **2)

out = np.load("/Users/masatakaaizawa/github/Matrices.npz")
W = out["W"] ## weight matrix
lc = out["O"][:,0] ## light curve 
P = out["P"] ## PC2 at each pixel
nside=16
lc_plus = lc - np.min(lc) + 0.005
sigma_arr = np.zeros(len(lc)) + 1



## Save W matrix
W_mean = np.mean(W, axis = 0)
hp.mollview(W_mean, title="Weight matrix",flip="geo",cmap=plt.cm.bone,min=0,max=np.max(W_mean))
hp.graticule(color="white");
plt.savefig( folder_root_result + folder_branch + "/others/" + "weight_mean.pdf", dpi=200, bbox_inches="tight")
plt.close()


## Making Constructor for CV with sparse modeling

neighbor_matrix, test = calc_neighbor_weightmatrix(hp, nside)
main_analyze = main_sparse_healpix(lc_plus, W, neighbor_matrix, sigma_arr, hp)
main_analyze.maxiter = 1000


## Combination of (l1, ltsv) for CV 

l1_min, l1_max = int(l1_parser_min) , int(l1_parser_max)
ltsv_min, ltsv_max = int(ltsv_parser_min), int(ltsv_parser_max)
l1_arr = [10**i for i in np.arange(l1_min, l1_max+one_step_for_l1, one_step_for_l1)]
ltsv_arr = [10**i for i in np.arange(ltsv_min, ltsv_max + one_step_for_ltsv, one_step_for_ltsv)]

## Main part of CV (cross validation)

folder_out = folder_root_result + folder_branch + "/"
if args.cv:
    logger.info("Start CV")
    main_analyze.cv(l1_arr,ltsv_arr, n_fold=10,  folder_name = folder_out, _debug = args.debug,index_generator = main_analyze.lineary_divided_index)
else:
    logger.info("Start solver w/o CV")
    main_analyze.solve_without_cv(l1_arr,ltsv_arr, folder_name = folder_out, _debug = args.debug)


## Tikhonov Solver
#tikhonov_result_out(W, lc, len(W), folder_name = folder_out)

