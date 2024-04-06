#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <mpi.h>
#include <omp.h>

#include "pgm.h"
#include "viz.h"
#include "log.h"
#include "types.h"

#define TAG_BCAST_DONE 1337
#define TAG_TASK_READY 7
#define TAG_TASK_ROW 8
#define TAG_TASK_ROW_RESULT 9

const int DONE = 1;

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

  llog(4, "nx = %d, ny = %d, xL = %.5f, yL = %.5f, xR = %.5f, yR = %.5f, Imax = %d\n", nx, ny, xL, yL, xR, yR, Imax);

  dx = (double) (xR - xL) / (double) (nx - 1);
  llog(4, "dx = %.2f\n", dx);
  dy = (double) (yR - yL) / (double) (ny - 1);
  llog(4, "dy = %.2f\n", dy);

  llog(4, "nx * ny = %d\n", nx * ny);
  matrix = (mb_t*) malloc(sizeof(mb_t) * nx * ny);

  for (int i = 0; i < nx; i++) {
    x = xL + i * dx;
    // #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < ny; j++) {
      y = yL + j * dy;
      c = x + I * y;
      matrix[j * nx + i] = mandelbrot_func(0 * I, c, 0, Imax);
    }
  }

  // int rank;
  // char* partfile;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // partfile = (char*) malloc(sizeof(char) * 21);
  // sllog(4, partfile, "test%d.pgm", rank);

  // write_pgm_image(matrix, Imax, nx, ny, partfile);
  llog(4, "matrix[0] = %d\n", matrix[0]);
  llog(4, "matrix[1] = %d\n", matrix[1]);

  // free(partfile);
  return matrix;
}

mb_t* _mandelbrot_matrix_row(int r, int nx, int ny, double xL, double yL, double xR, double yR, int Imax)
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
  matrix = (mb_t*) malloc(sizeof(mb_t) * nx);

  // #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < nx; i++) {
    x = xL + i * dx;
    // #pragma omp parallel for schedule(dynamic)
    for (int j = r; j < r + 1; j++) {
      y = yL + j * dy;
      c = x + I * y;
      matrix[i] = mandelbrot_func(0 * I, c, 0, Imax);
#ifdef VIZ
      viz_render_pixel(matrix[i], r, i, Imax);
#endif
    }
  }

  return matrix;
}

// mb_t* mandelbrot_matrix(int nx, int ny, double xL, double yL, double xR, double yR, int Imax)
// {
//   mb_t* M;
//   mb_t* M_whole;
//   mb_t* MT;
//   int rank, size, dw, dh;
//   int* recvcounts;
//   int* displs;
//   double dx, dy;

//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &size);

//   dx = (double) (xR - xL) / (double) size;
//   dy = (double) (yR - yL) / (double) size;
//   dw = (int) floor((double) nx / (double) size);
//   dh = (int) floor((double) ny / (double) size);
  
//   llog(4, "dx = %.5f, rank = %d, dh = %d, size = %d\n", dx, rank, dh, size);
//   M = _mandelbrot_matrix(
//     nx,
//     rank == size - 1 ? dh + (ny % size) : dh,
//     xL,
//     yL + rank * dy,
//     xR,
//     rank == size - 1 ? yR : yL + (rank + 1) * dy,
//     Imax
//   );
//   // M = _mandelbrot_matrix(
//   //   rank == size - 1 ? dw + (nx % size) : dw,
//   //   ny,
//   //   xL + rank * dx,
//   //   yL,
//   //   rank == size - 1 ? xR : xL + (rank + 1) * dx,
//   //   yR,
//   //   Imax
//   // );
//   if (rank == 0) {
//     M_whole = (mb_t*) malloc(nx * ny * sizeof(mb_t));
//     recvcounts = (int*) malloc(sizeof(int) * size);
//     displs = (int*) malloc(sizeof(int) * size);
//     for (int r = 0; r < size; r++) {
//       recvcounts[r] = ((r == size - 1) ? dh + (ny % size) : dh) * nx;
//       displs[r] = r == 0 ? 0 : displs[r - 1] + recvcounts[r - 1];
//       llog(4, "recvcounts[%d] = %d, displs[%d] = %d\n", r, recvcounts[r], r, displs[r]);
//     }
//   }
//   // if (rank == 0) {
//   //   M_whole = (mb_t*) malloc(nx * ny * sizeof(mb_t));
//   //   recvcounts = (int*) malloc(sizeof(int) * size);
//   //   displs = (int*) malloc(sizeof(int) * size);
//   //   for (int r = 0; r < size; r++) {
//   //     recvcounts[r] = ((r == size - 1) ? dw + (nx % size) : dw) * ny;
//   //     displs[r] = r == 0 ? 0 : displs[r - 1] + recvcounts[r - 1];
//   //     llog(4, "recvcounts[%d] = %d, displs[%d] = %d\n", r, recvcounts[r], r, displs[r]);
//   //   }
//   // }
//   llog(4, "dw * ny = %d\n", dw * ny);

//   // MPI_Gatherv(M, (rank == size - 1 ? (dh + (ny % size)) : dh) * nx, MPI_UNSIGNED_SHORT, M_whole, recvcounts, displs, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
//   MPI_Gatherv(M, (rank == size - 1 ? (dw + (nx % size)) : dw) * ny, MPI_UNSIGNED_SHORT, M_whole, recvcounts, displs, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);

//   // MT = (mb_t*) malloc(nx * ny * sizeof(mb_t));
//   // for (int i = 0; i < nx; i++) {
//   //   for (int j = 0; j < ny; j++) {
//   //     MT[i * ny + j] = M_whole[j * nx + i];
//   //   }
//   // }

//   if (rank == 0) {
//     // for (int i = 0; i < size; i++) {
//     //   llog(4, "rank=%d M[0] = %d\n", i, M_whole[displs[i]]);
//     //   llog(4, "rank=%d M[1] = %d\n", i, M_whole[displs[i] + 1]);
//     // }
//     free(recvcounts);
//     free(displs);
//   }
  
//   return M;
// }

mb_t* mandelbrot_matrix_rr(int nx, int ny, double xL, double yL, double xR, double yR, int Imax)
{
  int size, rank, done = 0;
  // int resolved_request_idx;
  MPI_Status status;
  MPI_Request* recv_requests;
  mb_t* M;
  // struct timeval start, end;
  // long microseconds = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank > 0) {
    // signal root that this rank is available to receive a new task to carry out
    M = (mb_t*) malloc(nx * sizeof(mb_t));
    MPI_Send(M, nx, MPI_UNSIGNED_SHORT, 0, TAG_TASK_ROW_RESULT, MPI_COMM_WORLD);
    llog(4, "[rank %d] sent dummy result\n", rank);

    while (!done) {
      recv_requests = (MPI_Request*) malloc(sizeof(MPI_Request) * 2);
      int requested_row;

      // setup receival of the completion message, sent by root when all due work is done
      // MPI_Irecv(&done, 1, MPI_INT, 0, TAG_BCAST_DONE, MPI_COMM_WORLD, recv_requests);

      // setup receival of a task to be performed
      // MPI_Irecv(&requested_row, 1, MPI_INT, 0, TAG_TASK_ROW, MPI_COMM_WORLD, recv_requests + 1);
      MPI_Recv(&requested_row, 1, MPI_INT, 0, TAG_TASK_ROW, MPI_COMM_WORLD, &status);

      // llog(4, "[rank %d] entering barrier\n", rank);
      // MPI_Barrier(MPI_COMM_WORLD);

      // wait for either the completion message or a new task
      llog(2, "[rank %d] waiting for either a task or completion\n", rank);
      // MPI_Waitany(2, recv_requests, &resolved_request_idx, &status);
      if (requested_row == -1) {
        done = 1;
        llog(2, "[rank %d] all done here, terminating\n", rank);
      }

      // gettimeofday(&end, NULL);
      // if (microseconds == -1) {
      //   microseconds = 0;
      // } else {
      //   microseconds += end.tv_usec - start.tv_usec;
      //   llog(4, "[rank %d] waiting time: %06ld microseconds\n", rank, microseconds);
      // }

      if (status.MPI_TAG == TAG_TASK_ROW && requested_row >= 0) {
        llog(4, "[rank %d] computing row %d\n", rank, requested_row);
        // perform the actual computation of the requested row
        M = _mandelbrot_matrix_row(
          requested_row,
          nx,
          ny,
          xL,
          yL,
          xR,
          yR,
          Imax
        );
        // send the result back to root
        llog(2, "[rank %d] sending back to root\n", rank);
        MPI_Send(M, nx, MPI_UNSIGNED_SHORT, 0, TAG_TASK_ROW_RESULT, MPI_COMM_WORLD);

        // gettimeofday(&start, NULL);
        free(recv_requests);
      }
    }
  } else {
    int nrow, next_row, recvd_rows = 0;
    int* assigned_rows;
    MPI_Status status;
    mb_t* row;

    assigned_rows = (int*) malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++) {
      assigned_rows[i] = -1;
    }

    next_row = 0;
    // M = (mb_t*) malloc(sizeof(mb_t) * nx * ny);
    M = (mb_t*) calloc(nx * ny, sizeof(mb_t));
    row = (mb_t*) malloc(sizeof(mb_t) * nx);
    while (recvd_rows < ny) {
      // wait for any rank to be ready to receive a new task (row to be computed)
      MPI_Recv(row, nx, MPI_UNSIGNED_SHORT, MPI_ANY_SOURCE, TAG_TASK_ROW_RESULT, MPI_COMM_WORLD, &status);

      nrow = assigned_rows[status.MPI_SOURCE];
      if (nrow != -1) {
        memcpy(M + nrow * nx, row, nx * sizeof(mb_t));
#ifdef VIZ
        viz_render(M, nx, ny, Imax);
#endif
        recvd_rows++;
      }

      // llog(4, "[rank %d] entering barrier\n", rank);
      // MPI_Barrier(MPI_COMM_WORLD);

      // send the ready rank some work to do, i.e. the next available row to be computed
      if (next_row < ny) {
        llog(4, "assigning row %d to rank %d\n", next_row, status.MPI_SOURCE);
        MPI_Send(&next_row, 1, MPI_INT, status.MPI_SOURCE, TAG_TASK_ROW, MPI_COMM_WORLD);
        assigned_rows[status.MPI_SOURCE] = next_row;
        next_row++;
      }
    }

    // MPI_Request* done_requests = (MPI_Request*) malloc(sizeof(MPI_Request) * size);
    next_row = -1;
    for (int i = 1; i < size; i++) {
      // MPI_Isend(&DONE, 1, MPI_INT, i, TAG_BCAST_DONE, MPI_COMM_WORLD, done_requests + i);
      // MPI_Send(&DONE, 1, MPI_INT, i, TAG_BCAST_DONE, MPI_COMM_WORLD);
      MPI_Send(&next_row, 1, MPI_INT, i, TAG_TASK_ROW, MPI_COMM_WORLD);
    }
    // MPI_Waitall(size, done_requests, MPI_STATUSES_IGNORE);
    llog(2, "all done\n");
  }
  // llog(4, "[rank %d] total waiting time: %06ld microseconds\n", rank, microseconds);
  return M;
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

  /* We need both OpenMPI and MPI to run properly: check for the OMPI_COMM_WORLD_SIZE 
     environment variable, which is set by mpirun on start. */
  char* world_size;
  world_size = getenv("OMPI_COMM_WORLD_SIZE");
  if (world_size == NULL) {
    llog(4, "Error: it seems that the program was not run with mpirun. Please run with: mpirun [options] %s\n", argv[0]);
    exit(1);
  }

  /* Initialize MPI using the MPI_THREAD_FUNNELED threading option, which allows only
     the master thread in every process (rank) to perform MPI calls. */
  int mpi_thread_init; 
  int rank, size;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_init); 
  if (mpi_thread_init < MPI_THREAD_FUNNELED) { 
    llog(4, "Error: could not initialize MPI with MPI_THREAD_FUNNELED\n");
    MPI_Finalize(); 
    exit(1); 
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef VIZ
  if (rank == 0) {
    if (viz_init("Mandelbrot", nx, ny)) {
      llog(4, "Error: could not initialize SDL\n");
      exit(1);
    }
  }
#endif
  
  // if (size > 1) {
  //   M = mandelbrot_matrix(nx, ny, xL, yL, xR, yR, Imax);
  // } else {
  //   M = _mandelbrot_matrix(nx, ny, xL, yL, xR, yR, Imax);
  // }
  // for (int i = 0; i < nx; i++) {
  //   for (int j = 0; j < ny; j++) {
  //     llog(4, "%d ", M[i * ny + j]);
  //   }
  //   llog(4, "\n");
  // }
  M = mandelbrot_matrix_rr(nx, ny, xL, yL, xR, yR, Imax);

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
  //   llog(4, "I am thread %d on rank %d\n", thread_id, rank);
  // }

#ifdef VIZ
  viz_destroy();
#endif
  MPI_Finalize(); 
  return 0; 
}