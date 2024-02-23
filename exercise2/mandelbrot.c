#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include <mpi.h>
#include <omp.h>

#include "pgm.h"

#define mb_t char

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

mb_t* mandelbrot_matrix(int nx, int ny, double xL, double yL, double xR, double yR, int Imax)
{
  double dx, dy, x, y;
  double complex c;
  mb_t* matrix;

  dx = (double) (xR - xL) / (double) (nx - 1);
  printf("dx = %.2f\n", dx);
  dy = (double) (yR - yL) / (double) (ny - 1);
  printf("dy = %.2f\n", dy);

  matrix = (mb_t*) malloc(sizeof(mb_t) * nx * ny);
  for (int i = 0; i < nx; i++) {
    x = xL + i * dx;
    for (int j = 0; j < ny; j++) {
      y = yL + j * dy;
      c = x + I * y;
      matrix[j * nx + i] = mandelbrot_func(0 * I, c, 0, Imax);
    }
  }
  return matrix;
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
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_init); 
  if (mpi_thread_init < MPI_THREAD_FUNNELED) { 
    printf("Error: could not initialize MPI with MPI_THREAD_FUNNELED\n");
    MPI_Finalize(); 
    exit(1); 
  }

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
    printf(" I_max: iteration boundary before considering a candidate point to be part of the Mandelbrot set\n");
    printf(" output_image: filename of the image to be generated\n");
    exit(1);
  }

  int nx, ny, Imax;
  double xL, yL, xR, yR;
  char* image_name;

  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  xL = atof(argv[3]);
  yL = atof(argv[4]);
  xR = atof(argv[5]);
  yR = atof(argv[6]);
  Imax = atoi(argv[7]);
  image_name = argv[8];

  mb_t* M = mandelbrot_matrix(nx, ny, xL, yL, xR, yR, Imax);
  // for (int i = 0; i < nx; i++) {
  //   for (int j = 0; j < ny; j++) {
  //     printf("%d ", M[i * ny + j]);
  //   }
  //   printf("\n");
  // }

  write_pgm_image(M, 127, nx, ny, image_name);

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

  MPI_Finalize(); 
  return 0; 
}