from subprocess import call, DEVNULL, PIPE, Popen
import time
import os
import re

SIZE = 4096
IMAX = 65535
X_L = -3
Y_L = -3
X_R = 3
Y_R = 3

NNODES = 2
NSOCKETS = NNODES * 2
CORES_PER_SOCKET = 64

if __name__ == "__main__":
  header = f"nodes={NNODES} sockets={NSOCKETS} EPYC size = {SIZE} Imax = {IMAX} xL = {X_L} yL = {Y_L} xR = {X_R} yR = {Y_R}"
  print("*" * len(header))
  print(header)
  print("*" * len(header))

  cores = CORES_PER_SOCKET * NSOCKETS
  while cores > 0:
    start = time.time()
    call(["make", "run"], env={
      "ARGS": f"{SIZE} {SIZE} {X_L} {Y_L} {X_R} {Y_R} {IMAX} test.pgm",
      "np": str(cores),
      "OMP_NUM_THREADS": "1",
      **os.environ
    }, stdout=DEVNULL, stderr=DEVNULL)
    end = time.time()

    elapsed = end - start
    print(f"cores={cores} | {elapsed:.3f}")

    cores //= 2