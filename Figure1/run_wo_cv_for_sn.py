import healpy as hp
import pylab 
import matplotlib.pyplot as plt
import time
import datetime
import logging
import parser
import argparse
import sys
import os

sys.path.append(os.getcwd()+"/../Spare_module/")
print(os.getcwd()+"/../Spare_module/")

from tsv_healpix import * 
from tikhonov_solve import *
from geometry import *
import log_make


#python l_curve_prep.py -root_f "./result_l_curve/" -f "data_for_lcurve_l1_-3.25_-3" -l1 -5 -0 -ltsv -3.25 -3 -step_ltsv 0.25 -step_l1 0.05

parser = argparse.ArgumentParser()

parser.add_argument("-root_f", default='./result/', help="root folder")
parser.add_argument("-f", default='cv', help="Name of folder including result")
parser.add_argument("-sn", default='20', type=float, help="Signal to noise ratio")
parser.add_argument("-l1", nargs = 2, default=[-4, 1], help="[l1min, l1max]")
parser.add_argument("-ltsv", nargs = 2, default=[-4, 1], help="[ltsvmin, ltsvmax]")
parser.add_argument("-step_l1",  type=float,default=1, help="step_l1")
parser.add_argument("-step_ltsv",type=float, default=1, help="step_ltsv")
parser.add_argument("-zeta",  default = np.pi/3.0,type=float, help="Obliquity of planet")
parser.add_argument("-teq",  default = np.pi, type=float,help="Equinox of planet")
parser.add_argument("-inc",  default = 0, type=float,help="Inclination")
parser.add_argument("-ndata", default = 1024,type=int,help="Equinox of planet")
parser.add_argument("-n_fold", default = 10,type=int,help="CV number")
parser.add_argument("-cv", action='store_true')
parser.add_argument("-debug", action='store_true')
parser.add_argument("-noneg", action='store_true')


args = parser.parse_args()
s_n_parser = args.sn
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

if args.debug:
    logger = log_make.log_make("main",log_folder, logging.INFO)
else:
    logger = log_make.log_make("main",log_folder, logging.DEBUG)

# Loading map (Npix = 12 * nside **2)
logger.info("Loading map")

nside=16
mmap=hp.read_map("../Data_mock/mockalbedo16.fits",verbose=False)
M=len(mmap)


# Setting geometry & system parameters (7 in total)
inc=0
Thetaeq=args.teq
zeta=args.zeta                                                                                             
Pspin=23.9344699/24.0 #Pspin: a sidereal day                                                                                                                                                                                                               
wspin=2*np.pi/Pspin                                                                                                          
Porb=365.242190402   #Porb=40.0
                        
N=int(args.ndata)


## Calculating lightcuve (lc) & transfer matrix (W)
folder_fig_lc = folder_root_result + folder_branch + "/others/"
lc, W, sigma_arr = light_curve_and_matrix(nside,zeta,inc,Thetaeq, Pspin, Porb, N, mmap, s_n_parser, folder_others = folder_fig_lc)

## Save W matrix
W_mean = np.mean(W, axis = 0)
hp.mollview(W_mean, title="Weight matrix",flip="geo",cmap=plt.cm.bone,min=0,max=np.max(W_mean))
hp.graticule(color="white");
plt.savefig( folder_root_result + folder_branch + "/others/" + "weight_mean.pdf", dpi=200, bbox_inches="tight")
plt.close()

## For calclulation of TSV in healpix
neighbor_matrix, test = calc_neighbor_weightmatrix(hp, nside)

## Choosing prox map
if args.noneg:
	prox_map = soft_threshold_nonneg_healpix
else:
	prox_map = soft_threshold_healpix

miniter= 2000
maxiter =4000

## Making Constructor for CV with sparse modeling
main_analyze = main_sparse_healpix(lc, W, neighbor_matrix, sigma_arr, hp, prox_map = prox_map,
	miniter = miniter, maxiter=maxiter)


## Combination of (l1, ltsv) for CV 
l1_min, l1_max = float(l1_parser_min) , float(l1_parser_max)
ltsv_min, ltsv_max = float(ltsv_parser_min), float(ltsv_parser_max)
l1_arr = [10**i for i in np.arange(l1_min, l1_max+one_step_for_l1, one_step_for_l1)]
ltsv_arr = [10**i for i in np.arange(ltsv_min, ltsv_max + one_step_for_ltsv, one_step_for_ltsv)]

## Main part of CV (cross validation)

folder_out = folder_root_result + folder_branch + "/"

if args.cv:
    logger.info("Start CV")
    print (int(args.n_fold))
    main_analyze.cv(l1_arr,ltsv_arr, n_fold = int(args.n_fold),  folder_name = folder_out, _debug = args.debug, index_generator = main_analyze.make_random_index)
else:
    logger.info("Start solver w/o CV")
    main_analyze.solve_without_cv(l1_arr,ltsv_arr, folder_name = folder_out, _debug = args.debug)


## Tikhonov Solver
lc_for_tikhonov = lc/sigma_arr
W_for_tikhonov = np.zeros(np.shape(W))
for i in range(len(lc)):
    W_for_tikhonov[i] = W[i]/sigma_arr[i]

tikhonov_result_out(W_for_tikhonov , lc_for_tikhonov, M, folder_name = folder_out,l2_min=1e-1, l2_max = 1e3,  num_l2 = 41)

