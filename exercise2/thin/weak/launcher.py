from subprocess import call, DEVNULL, PIPE, Popen
import time
import os
import re

SIZE = 2048
IMAX = 65535
X_L = -3
Y_L = -3
X_R = 3
Y_R = 3

NNODES = 1
NSOCKETS = NNODES * 2
CORES_PER_SOCKET = 12

if __name__ == "__main__":
  header = f"nodes={NNODES} THIN start_size={SIZE} Imax={IMAX} xL={X_L} yL={Y_L} xR={X_R} yR={Y_R}"
  print("*" * len(header))
  print(header)
  print("*" * len(header))

  imax = IMAX
  size = SIZE
  cores = CORES_PER_SOCKET

  while size > 0:
    start = time.time()
    call(["make", "run"], env={
      "ARGS": f"{SIZE} {size} {X_L} {Y_L} {X_R} {Y_R} {IMAX} test.pgm",
      "np": str(cores),
      "OMP_NUM_THREADS": "1",
      **os.environ
    }, stdout=DEVNULL, stderr=DEVNULL)
    end = time.time()

    elapsed = end - start
    print(f"cores={cores} | nx={SIZE} | ny={size} | {elapsed:.3f}")

    size -= 2