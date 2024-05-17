#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>

#include <omp.h>

#include "pgm.h"
#include "log.h"
#include "types.h"

#define TAG_BCAST_DONE 1337
#define TAG_TASK_READY 7
#define TAG_TASK_ROW 8
#define TAG_TASK_ROW_RESULT 9
#define DONE 1

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

mb_t* mandelbrot_matrix_single(int nx, int ny, double xL, double yL, double xR, double yR, int Imax)
{
  double dx, dy, x, y;
  double complex c;
  mb_t* matrix;

  llog(4, "nx = %d, ny = %d, xL = %.5f, yL = %.5f, xR = %.5f, yR = %.5f, Imax = %d\n", nx, ny, xL, yL, xR, yR, Imax);

  dx = (double) (xR - xL) / (double) (nx - 1);
  llog(4, "dx = %.2f\n", dx);
  dy = (double) (yR - yL) / (double) (ny - 1);
  llog(4, "dy = %.2f\n", dy);

  llog(4, "nx * ny = %d\n", nx * ny);
  matrix = (mb_t*) malloc(sizeof(mb_t) * nx * ny);

  // #pragma omp parallel for schedule(dynamic)
  // for (int a = 0; a < nx * ny; a++) {
  //   matrix[a] = 0;
  // }

  // #pragma omp parallel for schedule(dynamic)
  // for (int a = 0; a < nx * ny; a++) {
  //   int i = a / nx;
  //   int j = a % nx;
  //   double x = xL + i * dx;
  //   double y = yL + j * dy;
  //   complex c = x + I * y;
  //   matrix[a] = mandelbrot_func(0 * I, c, 0, Imax);
  // }

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < nx; i++) {
    x = xL + i * dx;
    #pragma omp parallel for schedule(dynamic)
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
  /* Check whether all needed arguments were provided on the command line, else print the
     help message. */
  if (argc < 9) {
    llog(4, "Mandelbrot set: a hybrid parallel/distributed implementation (based on OpenMP + MPI)\n");
    llog(4, "\n");
    llog(4, "Usage: %s n_x n_y x_L y_L x_R y_R I_max output_image\n", argv[0]);
    llog(4, "\n");
    llog(4, "Parameters description:\n");
    llog(4, " n_x: width of the output image\n");
    llog(4, " n_y: height of the output image\n");
    llog(4, " x_L and y_L: components of the complex c_L = x_L + iy_L, bottom left corner of the considered portion of the complex plane\n");
    llog(4, " x_R and y_R: components of the complex c_R = x_R + iy_R, top right corner of the considered portion of the complex plane\n");
    llog(4, " Imax: iteration boundary before considering a candidate point to be part of the Mandelbrot set\n");
    llog(4, " output_image: filename of the image to be generated\n");
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

  llog(4, "max_mb_t = %d\n", max_mb_t);
  if (Imax > max_mb_t) {
    llog(4, "Error: Imax too large (%d > %d)\n", Imax, max_mb_t);
    exit(1);
  }

  M = mandelbrot_matrix_single(nx, ny, xL, yL, xR, yR, Imax);

  write_pgm_image(M, Imax, nx, ny, image_name);
  return 0; 
}
