#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>

#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9*(double)t.tv_nsec)
/** return time in second
*/
__host__ double get_elapsedtime(void)
{
  struct timespec st;
  int err = gettime(&st);
  if (err !=0) return 0;
  return (double)st.tv_sec + get_sub_seconde(st);
}

__host__ void init(double* A, double* B, double* C, int size)
{
  int i = 0, j = 0;

  srand(2019);

  for(i = 0; i < size; i++)
  {
    for(j = 0; j < size; j++)
    {
      A[i * size + j] = rand();
      B[i * size + j] = rand();
      C[i * size + j] = 0.0;
    }
  }
}

void mult(double* A, double* B, double* C, int size)
{
  int i = 0, j = 0, k = 0;

  for(i = 0; i < size; i++)
  {
    for(j = 0; j < size; j++)
    {
      double sum = 0.;
      for(k = 0; k < size; k++)
      {
        sum += A[i * size + k] * B[k * size + j];
      }
      C[i * size + j] = sum;
    }
  }
}

__global__ void kernel(double* A, double* B, double* C, int N)
{
	int x, y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	int id = x * N + y;
	
	C[0] = 9999999;
	double sum = 0.;

	for (int i = 0; i < N; ++i)
	{
		sum += A[x * N + i] * B[i * N + y];
	}

	C[id] = sum;
}

int main(int argc, char** argv){
  int N = 0;

  double *A_h = NULL;
  double *B_h = NULL;
  double *C_h = NULL;

  //double t0 = 0., t1 = 0., duration = 0.;

  N = (argc < 2)?1000:atoi(argv[1]);
  fprintf(stdout, "Matrix Multiplication\n  Size: %dx%d\n", N, N);

  dim3 DimBlock(32, 32, 1);
  dim3 DimGrid(ceil(N/32.), ceil(N/32.), 1);

  // Memory allocation
  A_h = (double*) malloc(sizeof(double) * N * N);
  B_h = (double*) malloc(sizeof(double) * N * N);
  C_h = (double*) malloc(sizeof(double) * N * N);

  // Value initialization
  init(A_h, B_h, C_h, N);

    //allocation des vecteurs sur le GPU avec gestion d'erreur : abort
  cudaError_t error;

  double *A_d = NULL;
  double *B_d = NULL;
  double *C_d = NULL;

  error = cudaMalloc((void**)&A_d, N*N*sizeof(double));
  if(error != cudaSuccess)
  	abort();

  error = cudaMalloc((void**)&B_d, N*N*sizeof(double));
  if(error != cudaSuccess)
  	abort();

  error = cudaMalloc((void**)&C_d, N*N*sizeof(double));
  if(error != cudaSuccess)
  	abort();
  //transfert des données de host a device (avec gestion d'erreur, bien sur)
  error = cudaMemcpy(A_d, A_h, N*N*sizeof(double), cudaMemcpyHostToDevice);
  if(error != cudaSuccess)
  	abort();

  error = cudaMemcpy(B_d, B_h, N*N*sizeof(double), cudaMemcpyHostToDevice);
  if(error != cudaSuccess)
  	abort();

  error = cudaMemcpy(C_d, C_h, N*N*sizeof(double), cudaMemcpyHostToDevice);
  if(error != cudaSuccess)
  	abort();

  for(int i=0; i<N; i++)
  {
	printf("%lf ", C_h[i]);
  }

  float time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  printf("%d %d %d\n%d %d %d", DimGrid.x, DimGrid.y, DimGrid.z, DimBlock.x, DimBlock.y, DimBlock.z);
  kernel<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, N);
  cudaMemcpy(C_h, C_d, N*N*sizeof(double), cudaMemcpyDeviceToHost); 

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Time to generate:  %3.1f ms \n", time);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  for(int i = 0;i < N; i++)
    {
	printf("%lf ", C_h[i]);
    }
  // Compute multiplication
  //t0 = get_elapsedtime();
  //mult(A_h, B_h, C_h, N);
  //t1 = get_elapsedtime();

  // Pretty print
  //duration = (t1 - t0);
  //uint64_t nb_op = N * N * N;
  //fprintf(stdout, "Performance results: \n");
  //fprintf(stdout, "  Time: %lf s\n", duration);
  //fprintf(stdout, "  MFlops: %.2f\n", (nb_op / duration)*1E-6);

  return 0;
}
