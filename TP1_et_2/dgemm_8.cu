#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_WIDTH 32
#define TAILLE 2048

#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9*(double)t.tv_nsec)
/** return time in second
*/
double get_elapsedtime(void)
{
  struct timespec st;
  int err = gettime(&st);
  if (err !=0) return 0;
  return (double)st.tv_sec + get_sub_seconde(st);
}

void init(double** A, double** B, double** C, int size)
{
  int i = 0, j = 0;

  srand(2019);

  for(i = 0; i < size; i++)
  {
    for(j = 0; j < size; j++)
    {
      A[i][j] = rand();
      B[i][j] = rand();
      C[i][j] = 0.0;
    }
  }
}

void mult(double** A, double** B, double** C, int size)
{
  int i = 0, j = 0, k = 0;

  for(i = 0; i < size; i++)
  {
    for(j = 0; j < size; j++)
    {
      double sum = 0.;
      for(k = 0; k < size; k++)
      {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

// QUESTION 4
__global__
void MulMatrixKernel(double* A, double* B, double* C, int N)
{

  __shared__ double share_A[BLOCK_WIDTH][BLOCK_WIDTH];
  __shared__ double share_B[BLOCK_WIDTH][BLOCK_WIDTH];
  double sum = 0;

  int col    = threadIdx.x + blockDim.x * blockIdx.x;
  int line  = threadIdx.y + blockDim.y * blockIdx.y;

  for(int tile = 0; tile < gridDim.x; tile++)
  {
    share_A[threadIdx.x][threadIdx.y] = A[line * N + tile * blockDim.x + threadIdx.x];
    share_B[threadIdx.x][threadIdx.y] = B[(tile*blockIdx.y + threadIdx.y) * N + col];
    __syncthreads();

    for(int i = 0; i < BLOCK_WIDTH; i++)
    {
	sum += share_A[i][threadIdx.y]*share_B[threadIdx.x][i];
    }
    __syncthreads();
  }

  C[line * N + col] = sum;

}
// FIN QUESTION 4

int main(int argc, char** argv){
  int N, i;

  double *A_data;
  double *B_data;
  double *C_data;

  double **A;
  double **B;
  double **C;

  double t0 = 0., t1 = 0., duration = 0.;

  N = (argc < 2)?TAILLE:atoi(argv[1]);
  fprintf(stdout, "Matrix Multiplication\n  Size: %dx%d\n", N, N);

  // Memory allocation
  A_data = (double*) malloc(sizeof(double) * N * N);
  B_data = (double*) malloc(sizeof(double) * N * N);
  C_data = (double*) malloc(sizeof(double) * N * N);

  A = (double**) malloc(sizeof(double *) * N);
  B = (double**) malloc(sizeof(double *) * N);
  C = (double**) malloc(sizeof(double *) * N);

  for(i = 0; i < N; i++)
  {
    A[i] = &A_data[i * N];
    B[i] = &B_data[i * N];
    C[i] = &C_data[i * N];
  }

  // Value initialization
  init(A, B, C, N);

  // QUESTION 8
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //FIN QUESTION 8

  // QUESTION 1
  double *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(double) * N * N);
  cudaMalloc(&d_B, sizeof(double) * N * N);
  cudaMalloc(&d_C, sizeof(double) * N * N);
  // FIN QUESTION 1

  // QUESTION 2
  cudaMemcpy(d_A, A_data, sizeof(double) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B_data, sizeof(double) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C_data, sizeof(double) * N * N, cudaMemcpyHostToDevice);
  // FIN QUESTION 2

  // QUESTION 3
  int nbBlocks = N / BLOCK_WIDTH;
  if(N % BLOCK_WIDTH) nbBlocks++;
  dim3 gridSize(nbBlocks, nbBlocks);
  dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
  // FIN QUESTION 3

  // QUESTION 4
  cudaEventRecord(start); // QUESTION 8
  MulMatrixKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop); // QUESTION 8
  // FIN QUESTION 4

  // QUESTION 5
  cudaMemcpy(C_data, d_C, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
  // FIN QUESTION 5

  // QUESTION 8
  cudaEventSynchronize(stop);
  uint64_t nb_op = N * N * N;
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Matrice %dx%d\n\tTemps: %f s\n\tMFlops: %.2f\n", N, N, milliseconds/1000, (nb_op / (milliseconds/1000))*1E-6);
  // FIN QUESTION 8

  // Compute multiplication
  t0 = get_elapsedtime();
//  mult(A, B, C, N);
  t1 = get_elapsedtime();

  // Pretty print
  duration = (t1 - t0);
  fprintf(stdout, "Performance results: \n");
  fprintf(stdout, "  Time: %lf s\n", duration);
  fprintf(stdout, "  MFlops: %.2f\n", (nb_op / duration)*1E-6);

  return 0;
}
