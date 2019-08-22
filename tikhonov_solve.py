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



def tikhonov_result_out(W, lc, M, folder_name):

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    mprior=np.zeros(M) #prior
    lamb=0.1 # regularization parameter
    start = time.time()
    U,S,VT=np.linalg.svd(W)
    elapsed_time = time.time() - start                                                                                      
    logger.info("elapsed_time (SVD):%f [sec]", elapsed_time)
    nlcurve=30
    lmin=0.1
    lmax=100.0
    lamseq=np.logspace(np.log10(lmin),np.log10(lmax),num=nlcurve)

    modelnormseq=[]
    residualseq=[]
    curveseq=[]

    for lamb in lamseq:
        mest,dpre,residual,modelnorm,curv_lcurve=tikhonov_regularization(W,lc,mprior,U,VT,S,lamb)
        modelnormseq.append(modelnorm)
        residualseq.append(residual)
        curveseq.append(curv_lcurve)
        logger.debug("lambda:%f, curv_lcurve: %f", lamb, curv_lcurve)
    residualseq=np.array(residualseq)
    modelnormseq=np.array(modelnormseq)
    imax=np.argmax(curveseq)
    lamb=lamseq[imax]
    logger.info("Best lambda:%f", lamb)

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

    dpi = 200
    #best map
    output = io.StringIO()
    sys.stdout = output

    hp.mollview(mest_lbest, title="",flip="geo",cmap=plt.cm.bone,min=0,max=1.0)
    hp.graticule(color="white");
    plt.savefig(folder_name + "others/best_kw.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.debug("Fit finished")

    np.save(folder_name + "model/best_tik" , mest_lbest)


    # too small lambda
    hp.mollview(mest_ls, title="",flip="geo",cmap=plt.cm.bone)
    hp.graticule(color="white");
    plt.savefig(folder_name + "others/small_lam_kw.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
    np.save(folder_name + "model/too_small_tik" , mest_ls)


    # too large lambda
    hp.mollview(mest_ll, title="",flip="geo",cmap=plt.cm.bone)
    hp.graticule(color="white");
    plt.savefig(folder_name + "others/large_lam_kw.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
    np.save(folder_name + "model/too_large_tik" , mest_ll)
        
    sys.stdout = sys.__stdout__



