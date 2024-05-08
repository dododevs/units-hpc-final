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
OMP_PLACES = "cores"
PLACES_PER_NODE = 128

if __name__ == "__main__":
  header = f"nodes={NNODES} EPYC places={OMP_PLACES} places_per_node={PLACES_PER_NODE} size = {SIZE} Imax = {IMAX} xL = {X_L} yL = {Y_L} xR = {X_R} yR = {Y_R}"
  print("*" * len(header))
  print(header)
  print("*" * len(header))

  for node in range(NNODES, 0, -1):
    places = PLACES_PER_NODE
    while places > 0:
      start = time.time()
      call(["make", "run"], env={
        "ARGS": f"{SIZE} {SIZE} {X_L} {Y_L} {X_R} {Y_R} {IMAX} test.pgm",
        "np": str(node),
        "OMP_PLACES": OMP_PLACES,
        "OMP_NUM_THREADS": str(places),
        **os.environ
      }, stdout=DEVNULL, stderr=DEVNULL)
      end = time.time()

      elapsed = end - start
      print(f"nodes={node} | places={places} | {elapsed:.3f}")

      places //= 2