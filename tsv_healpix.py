import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import time
import collections
import math
from PIL import Image
import io
import sys
import logging
from geometry import *


logger = logging.getLogger("main").getChild("tsv")


def make_folders(folder_root_result, folder_branch):
    if not os.path.isdir(folder_root_result):
        os.mkdir(folder_root_result)

    folder_out = folder_root_result + folder_branch +"/"
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)

    folder_model = folder_out + "model/" 
    if not os.path.isdir(folder_model):
        os.mkdir(folder_model)

    folder_model_cv = folder_out + "model/cv/" 
    if not os.path.isdir(folder_model_cv):
        os.mkdir(folder_model_cv)

    folder_others = folder_out + "others/" 
    if not os.path.isdir(folder_others):
        os.mkdir(folder_others)

    folder_fig = folder_out + "figure_map/" 
    if not os.path.isdir(folder_fig):
        os.mkdir(folder_fig)

    ## Constructing Log
    log_folder = folder_out + "log/"
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)


## Cauculation of matrix to express neighboring pixels 
## This is used for calculation of TSV & d_TSV
def calc_neighbor_weightmatrix(hp, nside):
    nside_now = nside
    Npix = 12 * nside **2
    Neighbor_matrix = np.zeros((Npix,Npix))
    Weight_tsv_matrix = np.zeros((Npix,Npix))
    for i in range(Npix):
        neighbor = hp.get_all_neighbours(nside_now, i)
        for j in range(8):
            neighbor_ind = neighbor[j]
            if neighbor_ind == -1:
                continue
            Neighbor_matrix[i][neighbor_ind] = 1
            Weight_tsv_matrix[i][i] += 0.5
            Weight_tsv_matrix[i][neighbor_ind] -= 0.5
            Weight_tsv_matrix[neighbor_ind][i] -= 0.5
            Weight_tsv_matrix[neighbor_ind][neighbor_ind] += 0.5
    return Weight_tsv_matrix,Neighbor_matrix


## Calculation of TSV
def TSV_healpix(weight, image):
    return np.dot( image, np.dot(weight,  image))

## Calculation of d_TSV
def d_TSV_healpix(weight, image):
    return 2 * np.dot(weight,  image)

def F_TSV_healpix(data, W,  x_d, lambda_tsv, weight, sigma):
    data_dif = data -  np.dot(W, x_d)
    return (np.dot(data_dif, (1/sigma**2) *data_dif)/2)  + TSV_healpix(weight, x_d) *  lambda_tsv

def F_obs_healpix(data, W,  x_d, sigma):
    data_dif = data -  np.dot(W, x_d)
    return (np.dot(data_dif, (1/sigma**2) *data_dif)/2)

# Derivative of ||y-Ax||^2 + TSV (F_TSV)
##  np.dot(A.T, data_dif) is n_image vecgtor, d_TSV(x_d) is the n_image vecgtor or matrix

def dF_dx_healpix(data, W, x_d, lambda_tsv, weight, sigma):
    data_dif = -(data -  np.dot(W, x_d))
    return np.dot(W.T, (1/sigma**2) *data_dif) +lambda_tsv*  d_TSV_healpix(weight, x_d)

## Calculation of Q(x, y) (or Q(P_L(y), y)) except for g(P_L(y))
## x_d2 = PL(y) (xvec1)
## x_d = y (xvec2)

def calc_Q_part_healpix(data, W,  x_d2, x_d, df_dx, L, lambda_tsv, weight, sigma):
    Q_core = F_TSV_healpix(data, W, x_d, lambda_tsv, weight, sigma)
    Q_core += np.dot((x_d2 - x_d), df_dx) + 0.5 * L * np.dot(x_d2 - x_d, x_d2 - x_d)
    return Q_core


## Calculation of soft_thresholding (prox)

def soft_threshold_nonneg_healpix(x_d, eta):
    vec = np.zeros(len(x_d))
    for i in range(len(x_d)):
        if x_d[i] > 1:
            vec[i] = 1;
        elif x_d[i] > eta:
            vec[i] = x_d[i] - eta
        else:
            vec[i] = 0
    return vec

def light_curve_and_matrix(nside,zeta,inc,Thetaeq, Pspin, Porb, N, mmap, s_n, folder_others):

    logger.info("Parameter Setting")
    logger.info("inc:%f, Thetaeq:%f, zeta:%f", inc, Thetaeq, zeta)
    logger.info("Pspin:%f, Porb:%f, N_data:%d", Pspin, Porb, N)


    ### Calculating Kernel Function (W) & Light Curve (lc)
    logger.info("Calculating Kernel & Lightcurve")

    wspin = 2*np.pi/Pspin  
    worb = 2*np.pi/Porb  
    obst=np.linspace(0.0,Porb,N)

    Thetav=worb*obst
    Phiv=np.mod(wspin*obst,2*np.pi)
    WI,WV=comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
    W=WV[:,:]*WI[:,:]
    lc_original=np.dot(W,mmap)

    ## Adding noise to data (lc_original) with s/n = s_n_parser
    lc_mean = np.mean(lc_original)
    normalize_lc = (s_n**2)/lc_mean
    lc = np.random.poisson(lc_original * normalize_lc)/normalize_lc
    sigma_arr = np.sqrt(lc_original * normalize_lc)/normalize_lc

    logger.info("Mean: %f, Mean(sigma): %f, s_n:%f" % (np.mean(lc), np.mean(sigma_arr), s_n))

    logger.info("Plotting Lightcurve")
    ## Plotting and Saving lightcurve

    fig= plt.figure(figsize=(10,7.5))
    plt.plot(obst/obst[-1],lc/np.max(lc),lw=2,color="gray")
    plt.tick_params(labelsize=18)
    plt.ylabel("light intensity",fontsize=18)
    plt.xlabel("time [yr]",fontsize=18)
    plt.title("cloudless Earth",fontsize=18)
    plt.savefig(folder_others + "sotlc.png", bbox_inches="tight", pad_inches=0.0)
    plt.close()
    Theta_lc = np.array([Thetav,lc,sigma_arr])
    np.save(folder_others + "light_theta", Theta_lc)
    np.save(folder_others + "weight_W", W)


    with open(folder_others + "lightcurve.dat", mode='w') as f:
        for i in range(len(lc)):
            f.write("%f %f %f \n" % (Thetav[i], lc[i], sigma_arr[i]))

    return lc, W, sigma_arr

## Function for MFISTA
def mfista_func_healpix(I_init, d, A_ten, weight,sigma, lambda_l1= 1e2, lambda_tsv= 1e-8, L_init= 1e4, eta=1.1, maxiter= 10000, max_iter2=100, 
                    miniter = 100, TD = 30, eps = 1e-5, log_flag = False, do_nothing_flag = False):
    if do_nothing_flag == True:
        return I_init

    ## Initialization
    mu, mu_new = 1, 1
    y = I_init
    x_prev = I_init
    cost_arr = []
    L = L_init
    
    ## The initial cost function
    cost_first = F_TSV_healpix(d, A_ten, I_init, lambda_tsv, weight, sigma)
    cost_first += lambda_l1 * np.sum(I_init)
    cost_temp, cost_prev = cost_first, cost_first

    ## Main Loop until iter_now < maxiter
    ## PL_(y) & y are updated in each iteration
    for iter_now in range(maxiter):
        cost_arr.append(cost_temp)
        
        ##df_dx(y)
        df_dx_now = dF_dx_healpix(d, A_ten, y, lambda_tsv, weight, sigma) 
        
        ## Loop to estimate Lifshitz constant (L)
        ## L is the upper limit of df_dx_now
        for iter_now2 in range(max_iter2):
            
            y_now = soft_threshold_nonneg_healpix(y - (1/L) * df_dx_now, lambda_l1/L)
            Q_now = calc_Q_part_healpix(d, A_ten, y_now, y, df_dx_now, L,  lambda_tsv, weight, sigma)
            F_now = F_TSV_healpix(d, A_ten, y_now,lambda_tsv, weight, sigma)
            
            ## If y_now gives better value, break the loop
            if F_now <Q_now:
                break
            L = L*eta

        L = L/eta
        mu_new = (1+np.sqrt(1+4*mu*mu))/2
        F_now += lambda_l1 * np.sum(y_now)

        ## Updating y & x_k
        if F_now < cost_prev:
            cost_temp = F_now
            tmpa = (1-mu)/mu_new
            x_k = soft_threshold_nonneg_healpix(y - (1/L) * df_dx_now, lambda_l1/L)
            y = x_k + ((mu-1)/mu_new) * (x_k - x_prev) 
            x_prev = x_k
            
        else:
            cost_temp = F_now
            tmpa = 1-(mu/mu_new)
            tmpa2 =(mu/mu_new)
            x_k = soft_threshold_nonneg_healpix(y - (1/L) * df_dx_now, lambda_l1/L)
            y = tmpa2 * x_k + tmpa * x_prev       
            x_prev = x_k
            
        if(iter_now>miniter) and cost_arr[iter_now-TD]-cost_arr[iter_now]<cost_arr[iter_now]*eps:
            break

        mu = mu_new
    log_out_now = "l1:%d, ltsv:%d, Current iteration: %d/%d,  L: %f, cost: %f, cost_chiquare:%f, cost_l1:%f, cost_ltsv:%f" % (lambda_l1, lambda_tsv, iter_now, maxiter, L, cost_temp, F_obs_healpix(d, A_ten, y, sigma),lambda_l1 * np.sum(y),lambda_tsv * TSV_healpix(weight,y))
    if log_flag:
        logger.debug(log_out_now)
    return y


## Main class for imaging with sparse modeling
class main_sparse_healpix:

    def __init__(self, d, A_ten, weight, sigma, heal_py, eta = 1.1, maxiter= 10000, maxiter2=100, miniter = 100, TD = 50, eps = 1e-5):
        self.maxiter = maxiter
        self.maxiter2 = maxiter2
        self.miniter = miniter
        self.TD = TD
        self.eps = eps
        self.A = A_ten
        self.d = d
        self.eta = eta
        self.N_data, self.Npix = np.shape(A_ten)
        self.weight = weight
        self.sigma = sigma


        self.hp = heal_py
        self.MSE_result = []
        self.MSE_sigma_result = []
        self.l1_result = []
        self.ltsv_result= []
        self.chi_square = []
        self.l1_term = []
        self.ltsv_term = []
        
    def cost_evaluate(self, I_now, lambda_l1, lambda_tsv):
        print (np.shape(I_now))
        test = np.dot(self.A, I_now)
        return F_obs_healpix(self.d, self.A, I_now, self.sigma), lambda_tsv * TSV_healpix(self.weight, I_now), lambda_l1 * np.sum(I_now)
    

    def make_random_index(self, n_fold):
        random_index = np.arange(len(self.d), dtype = np.int64)
        np.random.shuffle(random_index)
        random_index_mod = random_index % n_fold
        return (random_index_mod)

    def save_data(self, arr, folder_name, file_name):

        np.save(folder_name + file_name, arr)


    def cv_result_out(self, file_out):

        file_now = open(file_out, "w")
        print("MSE, MSE_std, l1, ltsv, chi, l1_term, ltsv_term", file = file_now)
        for i in range(len(self.MSE_result)):
            print ("%e, %e, %e, %e, %e, %e, %e" % (self.MSE_result[i], self.MSE_sigma_result[i],self.l1_result[i], self.ltsv_result[i], self.chi_square[i], self.l1_term[i], self.ltsv_term[i]), file = file_now)
        file_now.close()


    def solve_and_make_figure(self, lambda_l1, lambda_tsv, file_name, _debug =False):
        I_init = np.ones(self.Npix)
        I_est = mfista_func_healpix(I_init, self.d , A_ten =self.A, weight = self.weight, sigma = self.sigma, eta = self.eta, 
                        lambda_tsv =lambda_tsv,lambda_l1= lambda_l1, maxiter = self.maxiter, log_flag = True, do_nothing_flag = _debug)
        output = io.StringIO()
        sys.stdout = output
        self.hp.mollview(I_est , title="",flip="geo",cmap=plt.cm.bone,min=0,max=1.0)
        self.hp.graticule(color="white");
        sys.stdout = sys.__stdout__
        plt.savefig(file_name, dpi = 200)
        plt.close()
        return I_est




    def cv(self, lambda_l1_row, lambda_tsv_row, n_fold, folder_name = "./", _debug = False):

        ## Initialization of result arrays
        self.MSE_result = []
        self.MSE_sigma_result = []
        self.l1_result = []
        self.ltsv_result= []
        self.chi_square = []
        self.l1_term = []
        self.ltsv_term = []


        for l1_now in lambda_l1_row:
            for ltsv_now in lambda_tsv_row:
                rand_index = self.make_random_index(n_fold)
                MSE_arr = []
        
                for i in range(n_fold):

                    I_init = np.ones(self.Npix)
  

                    ## Separating the data into taining data and test data
                    A_ten_for_est = self.A[rand_index != i,:]
                    d_for_est = self.d[rand_index != i]
                    A_ten_for_test = self.A[rand_index == i,:]
                    d_for_test = self.d[rand_index == i]

                    sigma_for_est = self.sigma[rand_index!=i]
                    sigma_for_test = self.sigma[rand_index == i]
                    t1 = time.time() 
                    I_est = mfista_func_healpix(I_init, d_for_est,  A_ten_for_est, self.weight, sigma_for_est, eta = self.eta, 
                        lambda_tsv = ltsv_now,lambda_l1= l1_now, maxiter = self.maxiter,miniter = 35, TD = 30,log_flag = False,  do_nothing_flag = _debug)
                    t2 = time.time() 
                    MSE_now = F_obs_healpix(d_for_test, A_ten_for_test, I_est, sigma_for_test)/(len(d_for_test)*1.0)
                    MSE_arr.append(MSE_now)
                    str_log_out = "%d/%d, l1: %d/%d, ltsv: %d/%d. elapsed time: %f" % (i, n_fold, np.log10(l1_now), np.log10(np.max(lambda_l1_row)), 
                        np.log10(ltsv_now), np.log10(np.max(lambda_tsv_row)), t2-t1)
                    logger.debug(str_log_out)

                    self.save_data(I_est, folder_name + "model/cv/", "l1_%d_ltsv_%d_ite_%d_nfold_%d" % (np.log10(l1_now), np.log10(ltsv_now), i, n_fold))


                self.MSE_result.append(np.mean(MSE_arr))
                self.MSE_sigma_result.append(np.std(MSE_arr))
                self.l1_result.append(l1_now)
                self.ltsv_result.append(ltsv_now)

                I_est = self.solve_and_make_figure(l1_now, ltsv_now, _debug = _debug,file_name =folder_name + "figure_map/" + "l1_%d_ltsv_%d.pdf" % (np.log10(l1_now), np.log10(ltsv_now)))
                self.save_data(I_est, folder_name + "model/", "l1_%d_ltsv_%d" % (np.log10(l1_now), np.log10(ltsv_now)))
                
                chi_square_now = F_obs_healpix(self.d, self.A, I_est, self.sigma)
                l1_term_now = l1_now * np.sum(I_est)
                ltsv_term_now = ltsv_now*TSV_healpix(self.weight, I_est)

                self.l1_result.append(l1_now)
                self.ltsv_result.append(ltsv_now)
                self.chi_square.append(chi_square_now)
                self.l1_term.append(l1_term_now )
                self.ltsv_term.append(ltsv_term_now)

                inf_message = "l1: %d/%d, ltsv: %d/%d, chi:%e, l1:%e, ltsv:%e" % (np.log10(l1_now), np.log10(np.max(lambda_l1_row)), np.log10(ltsv_now), np.log10(np.max(lambda_tsv_row)), chi_square_now, l1_term_now, ltsv_term_now)
                logger.info(inf_message)

        # Outputting result
        self.cv_result_out(folder_name + "others/cv_result.dat")

        logger.info("End of CV")




    def solve_without_cv(self, lambda_l1_row, lambda_tsv_row, folder_name = "./", print_status = True, file_out = "cv_result", _debug = False):

        self.l1_result = []
        self.ltsv_result= []
        self.chi_square = []
        self.l1_term = []
        self.ltsv_term = []

        for l1_now in lambda_l1_row:
            for ltsv_now in lambda_tsv_row:


                t1 = time.time() 
                I_est = self.solve_and_make_figure(l1_now, ltsv_now, _debug = _debug, file_name =folder_name + "figure_map/" + "l1_%d_ltsv_%d.pdf" % (np.log10(l1_now), np.log10(ltsv_now)))                
                self.save_data(I_est,  folder_name + "model/", "l1_%d_ltsv_%d" % (np.log10(l1_now), np.log10(ltsv_now)))

                t2 = time.time() 
                str_log_out = "l1: %d/%d, ltsv: %d/%d. elapsed time: %f" % (np.log10(l1_now), np.log10(np.max(lambda_l1_row)), 
                        np.log10(ltsv_now), np.log10(np.max(lambda_tsv_row)), t2-t1)      
                logger.debug(str_log_out)
                
                chi_square_now = F_obs_healpix(self.d, self.A, I_est, self.sigma)
                l1_term_now = l1_now * np.sum(I_est)
                ltsv_term_now = ltsv_now*TSV_healpix(self.weight, I_est)

                self.chi_square.append(chi_square_now)
                self.l1_term.append(l1_term_now )
                self.ltsv_term.append(ltsv_term_now)
                self.l1_result.append(l1_now)
                self.ltsv_result.append(ltsv_now)


        self.without_cv_result_out(folder_name + "others/noncv_result.dat")

        logger.info("End of solver w/o CV")




    def without_cv_result_out(self, file_out):

        file_now = open(file_out, "w")
        print("l1, ltsv, chi, l1_term, ltsv_term", file = file_now)
        for i in range(len(self.l1_result)):
            print ("%e, %e, %e, %e, %e" % (self.l1_result[i], self.ltsv_result[i], self.chi_square[i], self.l1_term[i], self.ltsv_term[i]), file = file_now)
        file_now.close()



        
"""
class CVPlotter_healpix(object):
    def __init__(self, nv, nh, L1_list, Ltsv_list):
        self.nh = nh
        self.nv = nv

        self.left_margin = 0.1
        self.right_margin = 0.1
        self.bottom_margin = 0.1
        self.top_margin = 0.1
        total_width = 1.0 - (self.left_margin + self.right_margin)
        total_height = 1.0 - (self.bottom_margin + self.top_margin)
        dx = total_width / float(self.nh)
        dy = total_height / float(self.nv)
        self.dx = min(dx, dy)
        self.dy = self.dx
        f = plt.figure(num='CVPlot', figsize=(10,10))
        plt.clf()
        left = self.left_margin
        bottom = self.bottom_margin
        height = self.dy * self.nv
        width = self.dx * self.nh
        outer_frame = plt.axes([left, bottom, width, height])
        outer_frame.set_xlim(-0.5, self.nh - 0.5)
        outer_frame.set_ylim(-0.5, self.nv - 0.5)
        outer_frame.set_xlabel('log$_{10}$($\\Lambda_{t}$)', fontsize = 26)
        outer_frame.set_ylabel('log$_{10}$($\\Lambda_{l}$)', fontsize = 26)
        outer_frame.tick_params(labelsize = 22)
        outer_frame.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(list(range(self.nh))))
        outer_frame.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(list(range(self.nv))))
        outer_frame.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: int(math.log10(Ltsv_list[int(x)]))))
        outer_frame.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: int(math.log10(L1_list[int(x)]))))
        
        self.L1_list = L1_list
        self.Ltsv_list = Ltsv_list
        
        self.axes_list = collections.defaultdict(dict)

    def make_cv_figure(self, folder_name, outfile, file_name_head=""):

        for j in range(self.nv):
            for i in range(self.nh-1,-1,-1):
                data_file_name = "%s/%sl1_%d_ltsv_%d.npy" % (folder_name, file_name_head, np.log10(self.L1_list[i]), np.log10(self.Ltsv_list[j]))
                self.plotimage(i, j, np.load(data_file_name))

        self.draw()
        self.savefig(outfile)
        plt.show()
        plt.close()


    def plotimage(self, row, column, data):
        left = self.left_margin + column * self.dx
        bottom = self.bottom_margin + row * self.dy
        height = self.dx
        width = self.dy
        nx, ny = data.shape
        a = plt.axes([left, bottom, width, height])

        a.imshow(data,cmap="gray")
        a.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        a.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        (x_cen, y_cen) =  (np.unravel_index(np.argmax(data), data.shape))
        (x_cen, y_cen) =  (np.unravel_index(np.argmax(data), data.shape))
#        a.set_xlim(121+width,121-width)
#        a.set_ylim(y_cen-width,y_cen+width)
        self.axes_list[row][column] = a

    def draw(self):
        plt.draw()


    def savefig(self, figfile):
        plt.savefig(figfile,  dpi=300)

"""

