## Taken from https://github.com/HajimeKawahara/exojupyter


import numpy as np
import healpy as hp
import pylab 
import matplotlib.pyplot as plt
import time
import os
import logging
import io
import sys

logger = logging.getLogger("main").getChild("tikhonov")
level_contour = 70
#Tikhonov regularization

def tikhonov_regularization(Dm,Ao,Smprior,U,VT,S,lambdatik,p=None):
    import numpy as np
    #should be np.array
    sumphi=0.0
    dxidlam=0.0

    Ndata=len(Ao)
    Mmodel=len(Smprior)
    Sm=np.zeros(Mmodel) 

    dv=[]
    for idata in range(0,Ndata):
        dv.append(Ao[idata]-np.inner(Dm[idata,:],Smprior))

    if p is None:
        p=min(Ndata,Mmodel)
    
    for i in range(0,p):
           phii=(S[i]**2/(S[i]**2+lambdatik**2))
           phij=(S[i]/(S[i]**2+lambdatik**2))
           dxidlam=dxidlam+(1.0-phii)*(phij**2)*(np.inner(U[:,i],dv)**2)
           Sm=Sm+phij*np.inner(U[:,i],dv)*VT[i,:]
           
    dxidlam=-4.0*dxidlam/lambdatik

    Sm=Sm+Smprior    
    Aoest=np.dot(Dm,Sm)
    residual=np.linalg.norm(Ao-Aoest)
    modelnorm=np.linalg.norm(Sm-Smprior)

    lam2=lambdatik**2
    rho=residual*residual
    xi=modelnorm*modelnorm
    curv_lcurve=-2.0*xi*rho/dxidlam*(lam2*dxidlam*rho+2.0*lambdatik*xi*rho+lam2*lam2*xi*dxidlam)/((lam2*lam2*xi*xi+rho*rho)**1.5)

    return Sm,Aoest,residual,modelnorm,curv_lcurve

def NGIM_regularization(Dm,Ao,Smprior,U,VT,S,lim):
    import numpy as np
    #should be np.array

    p = np.sum(S>lim)
    print("NGIM/TSVD: we regard values below ",lim," as zero.")
    print("p = ", p)
    sumphi=0.0

    Ndata=len(Ao)
    Mmodel=len(Smprior)
    Sm=np.zeros(Mmodel) 

    dv=[]
    for idata in range(0,Ndata):
        dv.append(Ao[idata]-np.inner(Dm[idata,:],Smprior))

    for i in range(0,p):
        phij=(1.0/S[i])
        Sm=Sm+phij*np.inner(U[:,i],dv)*VT[i,:]

    Sm=Sm+Smprior    
    Aoest=np.dot(Dm,Sm)
    residual=np.linalg.norm(Ao-Aoest)
    modelnorm=np.linalg.norm(Sm-Smprior)

    return Sm,Aoest,residual,modelnorm

def ComputeContours(P, Levels, file_save):
    
    NPix=hp.pixelfunc.get_map_size(P)
    NSide=hp.pixelfunc.npix2nside(NPix)
    
    Lon,Lat=hp.pixelfunc.pix2ang(NSide,np.arange(NPix,dtype=int),lonlat=True)
    Lon=(Lon+180)%360-180
    Cont=plt.tricontour(Lon,Lat,P.reshape(-1),levels=Levels,colors='k')
    plt.close()
    
    Cont_coll= Cont.collections[0].get_paths()
    hp.visufunc.mollview(P.reshape(-1),flip='geo',cmap='brg', title='(a)')
    hp.graticule()
    
    for iCont in range(len(Cont_coll)):
        hp.visufunc.projplot(Cont_coll[iCont].vertices[:,0],Cont_coll[iCont].vertices[:,1],lonlat=True,color='k')
    
    plt.savefig(file_save, dpi = 200)
    plt.close()


def tikhonov_result_out(W, lc, M, folder_name, l2_min=1e-3, l2_max = 1e0, num_l2 = 15):

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    mprior=np.zeros(M) + np.sum(lc)/np.sum(W)#prior
    start = time.time()
    U,S,VT=np.linalg.svd(W)
    elapsed_time = time.time() - start                                                                                      
    logger.info("elapsed_time (SVD):%f [sec]", elapsed_time)
    nlcurve=num_l2
    lmin=l2_min
    lmax=l2_max
    lamseq=np.logspace(np.log10(lmin),np.log10(lmax),num=nlcurve)

    modelnormseq=[]
    residualseq=[]
    curveseq=[]

    for (i, lamb) in enumerate(lamseq):
        logger.info("lamb:%e" % lamb)
        mest,dpre,residual,modelnorm,curv_lcurve=tikhonov_regularization(W,lc,mprior,U,VT,S,lamb)
        modelnormseq.append(modelnorm)
        residualseq.append(residual)
        curveseq.append(curv_lcurve)
        logger.debug("lambda:%f, curv_lcurve: %f", lamb, curv_lcurve)
        np.save(folder_name + "model/model_%d" % i , mest)

    residualseq=np.array(residualseq)
    modelnormseq=np.array(modelnormseq)
    imax=np.argmax(curveseq)
    lamb=lamseq[imax]
    logger.info("Best lambda:%f", lamb)

    np.save(folder_name + "others/curve" , curveseq)



    #define small and large lambda cases
    ismall=imax-7 

    if ismall < 0:
        ismall = 0

    ilarge=imax+9

    if ilarge > len(residualseq)-1:
        ilarge = len(residualseq)-1

    #plot a L-curve
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(121)
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.ylabel("Norm of Model",fontsize=12)
    pylab.xlabel("Norm of Prediction Error",fontsize=12)
    ax.plot(residualseq,modelnormseq,marker=".",c="gray")
    ax.plot([residualseq[imax]],[modelnormseq[imax]],marker="o",c="green")
    ax.plot([residualseq[ismall]],[modelnormseq[ismall]],marker="s",c="red")
    ax.plot([residualseq[ilarge]],[modelnormseq[ilarge]],marker="^",c="blue")
    plt.tick_params(labelsize=12)
    ax2 = fig.add_subplot(122)
    pylab.xscale('log')
    pylab.ylabel("Curvature",fontsize=12)
    pylab.xlabel("Norm of Prediction Error",fontsize=12)
    ax2.plot(residualseq,curveseq,marker=".",c="gray")
    ax2.plot([residualseq[imax]],[curveseq[imax]],marker="o",c="green")
    plt.tick_params(labelsize=12)
    plt.savefig(folder_name +"others/lcurve.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()

    #getting maps!
    mest_lbest,dpre,residual,modelnorm,curv_lcurve=tikhonov_regularization(W,lc,mprior,U,VT,S,lamb)
    mest_ls,dpre,residual,modelnorm,curv_lcurve=tikhonov_regularization(W,lc,mprior,U,VT,S,lamseq[ismall])
    mest_ll,dpre,residual,modelnorm,curv_lcurve=tikhonov_regularization(W,lc,mprior,U,VT,S,lamseq[ilarge])

    # best lambda
    ComputeContours(mest_lbest, [np.percentile(mest_lbest, level_contour)],folder_name + "others/best_kw.pdf")
    np.save(folder_name + "model/best_tik" , mest_lbest)

    # small lambda
    ComputeContours(mest_ls, [np.percentile(mest_ls, level_contour)],folder_name + "others/too_small_kw.pdf")
    np.save(folder_name + "model/too_small_tik" , mest_ls)

    # large lambda
    ComputeContours(mest_ll, [np.percentile(mest_ll, level_contour)],folder_name + "others/too_large_kw.pdf")
    np.save(folder_name + "model/too_large_tik" , mest_ll)


def make_random_index(n_len, n_fold):
    random_index = np.arange(n_len, dtype = np.int64)
    np.random.shuffle(random_index)
    random_index_mod = random_index % n_fold
    return (random_index_mod)   



def tikhonov_result_out_cv(W, lc, M, folder_name, n_fold=2, l2_min=1e-3, l2_max = 1e0, num_l2 = 15):

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    mprior=np.zeros(M) #prior

    nlcurve=num_l2
    lmin=l2_min
    lmax=l2_max
    lamseq=np.logspace(np.log10(lmin),np.log10(lmax),num=nlcurve)
    folder_others = folder_name + "others/"
    folder_model = folder_name + "model/"
    if not os.path.exists(folder_others):
        os.makedirs(folder_others)

    if not os.path.exists(folder_model):
        os.makedirs(folder_model)


    rand_index = make_random_index(len(lc), n_fold)
    MSE_arr = np.zeros((len(lamseq),n_fold))

    for n_count in range(n_fold):
        W_fold = W[rand_index != n_count,:]
        U_fold,S_fold,VT_fold=np.linalg.svd(W[rand_index != n_count,:]) 
        lc_fold = lc[rand_index != n_count]
        for (i, lamb) in enumerate(lamseq):
            print (n_count, lamb)

            mest,dpre,residual,modelnorm,curv_lcurve=tikhonov_regularization(W_fold,lc_fold,mprior,U_fold,VT_fold,S_fold,lamb)
           
            lc_est=np.dot(W[rand_index == n_count,:],mest)
            residual=np.linalg.norm(lc[rand_index == n_count]-lc_est)/np.sqrt(len(lc_est))
            lc_est_all=np.dot(W,mest)

            MSE_arr[i][n_count] = residual
            np.save(folder_model  + "model_%d" % i , mest)
            np.save(folder_model  + "model_%d_lc" % i, (lc, lc_est_all))

    np.save(folder_others + "lamseq", lamseq)
    np.save(folder_others + "mse", MSE_arr)
    return lamseq, MSE_arr

def tikhonov_result_cv_half(W, lc, M, folder_name, l2_min=1e-3, l2_max = 1e0, num_l2 = 15):

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    mprior=np.zeros(M) #prior

    nlcurve=num_l2
    lmin=l2_min
    lmax=l2_max
    lamseq=np.logspace(np.log10(lmin),np.log10(lmax),num=nlcurve)
    folder_others = folder_name

    half_index = np.zeros(len(lc))
    for i in range(len(lc)):
        if i >(len(lc)/2):
            half_index[i] = 1
    MSE_arr = np.zeros((len(lamseq),2))

    for j in range(2):

        W_fold = W[half_index== j,:]
        U_fold,S_fold,VT_fold=np.linalg.svd(W[half_index == j,:]) 
        lc_fold = lc[half_index == j]

        for (i, lamb) in enumerate(lamseq):
            print (j, lamb)
            mest,dpre,residual,modelnorm,curv_lcurve=tikhonov_regularization(W_fold,lc_fold,mprior,U_fold,VT_fold,S_fold,lamb)
            lc_est=np.dot(W[half_index != j,:],mest)
            residual=np.linalg.norm(lc[half_index!= j]-lc_est)**2
            MSE_arr[i][j] = residual
            lc_est_all=np.dot(W,mest)

            np.save(folder_others + "model/model_%d_%d" % (i, j) , mest)
            np.save(folder_others + "model/model_%d_%d_lc" % (i, j), (lc, lc_est_all))

    np.save(folder_others + "others/lamseq", lamseq)
    np.save(folder_others + "others/mse", MSE_arr)
    return lamseq, MSE_arr

def cv(self, lambda_2, n_fold, folder_name = "./", _debug = False, index_generator = make_random_index):

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
            rand_index = index_generator(len(self.d), n_fold)

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
                I_est = mfista_func_healpix(I_init, d_for_est,  A_ten_for_est, self.weight, sigma_for_est, eta = self.eta, L_init = self.L_init, 
                    lambda_tsv = ltsv_now,lambda_l1= l1_now, maxiter = self.maxiter, miniter = self.miniter, log_flag = False,  do_nothing_flag = _debug, prox_map = self.prox_map,prox_crit = self.crit, prox_crit_flag = self.crit_flag)
                t2 = time.time() 
                MSE_now = F_obs_healpix(d_for_test, A_ten_for_test, I_est, sigma_for_test)/(len(d_for_test)*1.0)
                MSE_arr.append(MSE_now)
                str_log_out = "%d/%d, l1: %.2f/%.2f, ltsv: %.2f/%.2f. elapsed time: %f" % (i, n_fold, np.log10(l1_now), np.log10(np.max(lambda_l1_row)), 
                    np.log10(ltsv_now), np.log10(np.max(lambda_tsv_row)), t2-t1)
                logger.info(str_log_out) 

                self.save_data(I_est, folder_name + "model/cv/", "l1_%.2f_ltsv_%.2f_ite_%d_nfold_%d" % (np.log10(l1_now), np.log10(ltsv_now), i, n_fold))





            I_est = self.solve_and_make_figure(l1_now, ltsv_now, _debug = _debug,file_name =folder_name + "figure_map/" + "l1_%.2f_ltsv_%.2f.pdf" % (np.log10(l1_now), np.log10(ltsv_now)))
            self.save_data(I_est, folder_name + "model/", "l1_%.2f_ltsv_%.2f" % (np.log10(l1_now), np.log10(ltsv_now)))
            
            chi_square_now = F_obs_healpix(self.d, self.A, I_est, self.sigma)
            l1_term_now = l1_now * np.sum(np.abs(I_est))
            ltsv_term_now = ltsv_now*TSV_healpix(self.weight, I_est)


            self.MSE_result.append(np.mean(MSE_arr))
            self.MSE_sigma_result.append(np.std(MSE_arr))
            self.l1_result.append(l1_now)
            self.ltsv_result.append(ltsv_now)
            self.chi_square.append(chi_square_now)
            self.l1_term.append(l1_term_now)
            self.ltsv_term.append(ltsv_term_now)

            inf_message = "l1:%.2f/%.2f, ltsv: %.2f/%.2f, chi:%e, l1:%e, ltsv:%e" % (np.log10(l1_now), np.log10(np.max(lambda_l1_row)), np.log10(ltsv_now), np.log10(np.max(lambda_tsv_row)), chi_square_now, l1_term_now, ltsv_term_now)
            logger.info(inf_message)

    # Outputting result
    self.cv_result_out(folder_name + "others/cv_result.dat")

    logger.info("End of CV")



