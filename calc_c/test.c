#include <stdio.h>
#define N  1000
#define Nx 200
#define Ny 200
#include <time.h>


// We find that "einsum" in numpy has the same speed as that of c
double a[N][Nx][Ny], b[Nx][Ny];
double sum[N];

int main(void){
 
  int i, j, k, term;
  clock_t start,end;

  start = clock();

  for (i=0; i<N; i++){
    for (j=0; j<Nx; j++){
        for ( k=0; k< Ny; k++){
            a[i][j][k] = i + j + k;
        }
    }
  }
  end = clock();
  printf("%.2f秒かかりました\n",(double)(end-start)/CLOCKS_PER_SEC);

  start = clock();

  for (j=0; j<Nx; j++){
    for ( k=0; k< Ny; k++){
         b[j][k] = j*k;
    }
  }
  for (i=0; i<N; i++){
    sum[i] = 0;
    for (j=0; j<Nx; j++){
        for ( k=0; k< Ny; k++){
            sum[i]+= a[i][j][k] * b[j][k];
        }
    }
  }
  end = clock();
  printf("%.2f秒かかりました\n",(double)(end-start)/CLOCKS_PER_SEC);

  return 0;
}