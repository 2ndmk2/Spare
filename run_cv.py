import module
from module import *
import importlib
importlib.reload(module)
import readpng as rpng

## Setting for data
## Nx;  number of data along "i"
## Ny: number of data  along "j"
#img=rpng.get_bwimg("./figure_make/fig2.png")
img=rpng.get_bwimg("./figure_make/che.png")
#img = 1-img

Nx, Ny = np.shape(img)
N_data = 1000

I_ans = np.zeros((Nx, Ny))


### Making answers 
## I_{i,j}:  matrix (Nx, Ny)         I_{i,j} = Ny * i + j

for i in range(Nx):
    for j in range(Ny):
        I_ans[i][j] = Ny * i  + j + (i-j)**3
        
I_ans = img
#Testing 1d-2d array conversion   I_{i,j} <-> vec_I
# vec_I; array (Nx*Ny)    vector corresponding to I_ans
# re_I: same as I_ans
# vec_I[j + Ny*x] = I_{0,0}   I_{0,1} ... 
# vec_I is returned to (re_I) via ravel() and reshape()

vec_I = I_ans.ravel()  
re_I = (vec_I.reshape(Nx, Ny))


## Data making
## random_generator: Function for making data
## d; array (N_data)      Observed data (sum of luminosity)
## g: tensor (N_data, Nx, Ny)      Weighting matrix in observation (
rand_now = random_generator(N_data, Nx, Ny)
d, A_ten=rand_now.make_data(I_ans,20) 
A_ten = np.array(A_ten)


## Testing tensor calculation "Sum_{j,k} (g_{i,j,k} * I_ans_{j,k} )"
## See d = g_{i,j,k}  I_ans_{j,k} 

#print ("Sum_{j,k} (A_ten_{i,j,k} I_ans_{j,k}) = ", np.einsum("ijk,jk->i", A_ten, I_ans))
#print ("d_{i}=", d)
#print ("They agree!!")

ana_main = main_sparse(d, A_ten)
l1_arr = [10**i for i in [-8, -2]]
ltsv_arr = [10**i for i in [-8, -2]]
n_fold = 5
ana_main.cv(l1_arr, ltsv_arr, n_fold, folder_name = "./cv_test")




