#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

#include <mpi.h>
#include <omp.h>

#include "pgm.h"

#define mb_t unsigned short
#define max_mb_t (int) pow(2, 8 * sizeof(mb_t)) - 1

mb_t mandelbrot_func(double complex z, double complex c, int n, int Imax)
{
  if (cabs(z) >= 2.0) {
    return (mb_t) n;
  }
  if (n >= Imax) {
    return (mb_t) 0;
  }
  return mandelbrot_func(z * z + c, c, n + 1, Imax);
}

mb_t* _mandelbrot_matrix(int nx, int ny, double xL, double yL, double xR, double yR, int Imax)
{
  double dx, dy, x, y;
  double complex c;
  mb_t* matrix;

  printf("nx = %d, ny = %d, xL = %.5f, yL = %.5f, xR = %.5f, yR = %.5f, Imax = %d\n", nx, ny, xL, yL, xR, yR, Imax);

  dx = (double) (xR - xL) / (double) (nx - 1);
  printf("dx = %.2f\n", dx);
  dy = (double) (yR - yL) / (double) (ny - 1);
  printf("dy = %.2f\n", dy);

  printf("nx * ny = %d\n", nx * ny);
  matrix = (mb_t*) malloc(sizeof(mb_t) * nx * ny);

  for (int i = 0; i < nx; i++) {
    x = xL + i * dx;
    // #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < ny; j++) {
      y = yL + j * dy;
      c = x + I * y;
      matrix[j * nx + i] = mandelbrot_func(0 * I, c, 0, Imax);
      // matrix[j * nx + j] = j * nx + j;
    }
  }
  printf("M[0] = %d\n", matrix[0]);
  printf("M[1] = %d\n", matrix[1]);
  return matrix;
}

mb_t* mandelbrot_matrix(int nx, int ny, double xL, double yL, double xR, double yR, int Imax)
{
  mb_t* M;
  int rank, size, dw, dh;
  int* recvcounts;
  int* displs;
  double dx, dy;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // dx = (double) (xR - xL) / (double) (size - 1);
  // dw = (int) floor((double) nx / (double) (size - 1));
  // printf("dx = %.5f, rank = %d, dw = %d, size = %d\n", dx, rank, dw, size);
  // if (rank > 0) {
  //   M = _mandelbrot_matrix(
  //     rank == size - 1 ? dw + nx % (size - 1) : dw,
  //     ny, 
  //     xL + (rank - 1) * dx,
  //     yL, 
  //     rank == size - 1 ? xR : xL + rank * dx,
  //     yR,
  //     Imax
  //   );
  // } else {
  //   M = (mb_t*) malloc(nx * ny * sizeof(mb_t));
  //   recvcounts = (int*) malloc(sizeof(int) * size);
  //   displs = (int*) malloc(sizeof(int) * size);
  //   recvcounts[0] = 0;
  //   displs[0] = 0;
  //   for (int r = 1; r < size; r++) {
  //     recvcounts[r] = r == size - 1 ? dw + nx % (size - 1) : dw;
  //     printf("recvcounts[%d] = %d\n", r, recvcounts[r]);
  //     displs[r] = displs[r - 1] + recvcounts[r - 1];
  //   }
  // }
  // printf("dw * ny = %d\n", dw * ny);


  // MPI_Gatherv(M, rank == 0 ? 0 : ((rank == size - 1 ? dw + nx % (size - 1) : dw)) * ny, MPI_UNSIGNED_SHORT, M, recvcounts, displs, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
  // // MPI_Gather(M, (rank == size - 1 ? dw + nx % (size - 1) : dw) * ny, MPI_UNSIGNED_SHORT, M, (rank == size - 1 ? dw + nx % (size - 1) : dw), MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);

  dx = (double) (xR - xL) / (double) size;
  dy = (double) (yR - yL) / (double) size;
  dw = (int) floor((double) nx / (double) size);
  dh = (int) floor((double) ny / (double) size);
  
  printf("dx = %.5f, rank = %d, dh = %d, size = %d\n", dx, rank, dh, size);
  M = _mandelbrot_matrix(
    // rank == size - 1 ? dw + (nx % size) : dw,
    nx,
    // ny,
    rank == size - 1 ? dh + (ny % size) : dh,
    // xL + rank * dx,
    xL,
    // yL,
    yL + rank * dy,
    // rank == size - 1 ? xR : xL + (rank + 1) * dx,
    xR,
    // yR,
    rank == size - 1 ? yR : yL + (rank + 1) * dy,
    Imax
  );
  if (rank == 0) {
    M = (mb_t*) malloc(nx * ny * sizeof(mb_t));
    recvcounts = (int*) malloc(sizeof(int) * size);
    displs = (int*) malloc(sizeof(int) * size);
    for (int r = 0; r < size; r++) {
      // recvcounts[r] = (r == size - 1 ? dw + (nx % size) : dw) * ny;
      recvcounts[r] = (r == size - 1 ? dh + (ny % size) : dh) * nx;
      displs[r] = r == 0 ? 0 : displs[r - 1] + recvcounts[r - 1];
      printf("recvcounts[%d] = %d, displs[%d] = %d\n", r, recvcounts[r], r, displs[r]);
    }
  }
  printf("dw * ny = %d\n", dw * ny);


  // MPI_Gatherv(M, (rank == size - 1 ? dw + (nx % size) : dw) * ny, MPI_UNSIGNED_SHORT, M, recvcounts, displs, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(M, (rank == size - 1 ? dh + (ny % size) : dh) * nx, MPI_UNSIGNED_SHORT, M, recvcounts, displs, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
  // MPI_Gather(M, (rank == size - 1 ? dw + nx % (size - 1) : dw) * ny, MPI_UNSIGNED_SHORT, M, (rank == size - 1 ? dw + nx % (size - 1) : dw), MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("M = ");
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        printf("%d ", M[j * nx + i]);
      }
      printf("\n");
    }
    free(recvcounts);
    free(displs);
  }
  
  return M;
}

int main(int argc, char** argv)
{
  /* We need both OpenMPI and MPI to run properly: check for the OMPI_COMM_WORLD_SIZE 
     environment variable, which is set by mpirun on start. */
  char* world_size;
  world_size = getenv("OMPI_COMM_WORLD_SIZE");
  if (world_size == NULL) {
    printf("Error: it seems that the program was not run with mpirun. Please run with: mpirun [options] %s\n", argv[0]);
    exit(1);
  }

  /* Initialize MPI using the MPI_THREAD_FUNNELED threading option, which allows only
     the master thread in every process (rank) to perform MPI calls. */
  int mpi_thread_init; 
  int rank, size;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_init); 
  if (mpi_thread_init < MPI_THREAD_FUNNELED) { 
    printf("Error: could not initialize MPI with MPI_THREAD_FUNNELED\n");
    MPI_Finalize(); 
    exit(1); 
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Check whether all needed arguments were provided on the command line, else print the
     help message. */
  if (argc < 9) {
    printf("Mandelbrot set: a hybrid parallel/distributed implementation (based on OpenMP + MPI)\n");
    printf("\n");
    printf("Usage: %s n_x n_y x_L y_L x_R y_R I_max output_image\n", argv[0]);
    printf("\n");
    printf("Parameters description:\n");
    printf(" n_x: width of the output image\n");
    printf(" n_y: height of the output image\n");
    printf(" x_L and y_L: components of the complex c_L = x_L + iy_L, bottom left corner of the considered portion of the complex plane\n");
    printf(" x_R and y_R: components of the complex c_R = x_R + iy_R, top right corner of the considered portion of the complex plane\n");
    printf(" Imax: iteration boundary before considering a candidate point to be part of the Mandelbrot set\n");
    printf(" output_image: filename of the image to be generated\n");
    exit(1);
  }

  int nx, ny, Imax;
  double xL, yL, xR, yR;
  char* image_name;
  mb_t* M;

  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  xL = atof(argv[3]);
  yL = atof(argv[4]);
  xR = atof(argv[5]);
  yR = atof(argv[6]);
  Imax = atoi(argv[7]);
  image_name = argv[8];

  printf("max_mb_t = %d\n", max_mb_t);
  if (Imax > max_mb_t) {
    printf("Error: Imax too large (%d > %d)\n", Imax, max_mb_t);
    exit(1);
  }

  if (size > 1) {
    M = mandelbrot_matrix(nx, ny, xL, yL, xR, yR, Imax);
  } else {
    M = _mandelbrot_matrix(nx, ny, xL, yL, xR, yR, Imax);
  }
  // for (int i = 0; i < nx; i++) {
  //   for (int j = 0; j < ny; j++) {
  //     printf("%d ", M[i * ny + j]);
  //   }
  //   printf("\n");
  // }

  if (rank == 0) {
    write_pgm_image(M, Imax, nx, ny, image_name);
  }

  // int thread_id;
  // int rank;

  // #pragma omp parallel private(thread_id)
  // {
  //   #pragma omp master
  //   {
  //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //   }
    
  //   thread_id = omp_get_thread_num();
  //   printf("I am thread %d on rank %d\n", thread_id, rank);
  // }

  free(M);
  MPI_Finalize(); 
  return 0; 
}