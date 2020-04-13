import os
import numpy as np
from multiprocessing import Pool


def one_run(zeta, teq, inc, sn = 20):

	outfile = "non_cv_for_sn_zeta%d_teq%d_inc%d_sn_%d" % (zeta, teq, inc, sn)

	zeta = zeta * np.pi/180.0
	teq = teq *np.pi /180.0
	inc = inc * np.pi / 180.0
	run_str = "python run_wo_cv_for_sn.py -root_f './result/' -f '%s' -inc %f -zeta %f -teq %f -sn %d -l1 -3 1 -ltsv -3 1 -step_ltsv 0.25 -step_l1 0.25" % (outfile, inc, zeta, teq, sn)
	
	#test code
	#run_str = "python run_cv.py -root_f './result_cv_for_sn_9_16/' -f '%s' -inc %f -zeta %f -teq %f -sn %d -l1 -4 -4 -ltsv -4 -4 -n_fold 2" % (outfile, inc, zeta, teq, sn)
	print (run_str)
	os.system(run_str)

def wrapper_onerun(args):
    return one_run(*args)

arr = []


zeta_arr =np.array([90])
teq_arr = np.array([180])
inc_arr = np.array([0])
sn_arr = np.array([2,5, 100])

for dmy in zeta_arr:
	for dmy1 in teq_arr:
		for dmy2 in inc_arr:
			for dmy3 in sn_arr:

				arr.append([dmy, dmy1, dmy2, dmy3])

p = Pool(processes=4)
p.map(wrapper_onerun, arr)