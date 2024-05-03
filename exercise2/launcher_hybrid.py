from subprocess import call, DEVNULL
import time
import os

SIZE = 1024
IMAX = 65535
X_L = -3
Y_L = -3
X_R = 3
Y_R = 3

MIN_PROCESSES = 1
MAX_PROCESSES = 2

MIN_PLACES = 1
MAX_PLACES = 128

if __name__ == "__main__":
  header = f"size = {SIZE} Imax = {IMAX} xL = {X_L} yL = {Y_L} xR = {X_R} yR = {Y_R}"
  print("*" * len(header))
  print(header)
  print("*" * len(header))

  for npl in range(MAX_PLACES, MIN_PLACES - 1, -1):
    for np in range(MAX_PROCESSES, MIN_PROCESSES - 1, -1):
      start = time.time()
      call(["make", "run"], env={
        "ARGS": f"{SIZE} {SIZE} {X_L} {Y_L} {X_R} {Y_R} {IMAX} test.pgm",
        "np": str(np),
        "OMP_PLACES": "cores",
        "OMP_NUM_THREADS": str(npl),
        **os.environ
      }, stdout=DEVNULL, stderr=DEVNULL)
      end = time.time()

      elapsed = end - start
      print(f"np={np} | {elapsed:.3f}")
