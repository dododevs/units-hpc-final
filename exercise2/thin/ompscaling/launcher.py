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

NNODES = 1
NSOCKETS = 1
OMP_PLACES = "cores"
OMP_PROC_BIND = "close"
PLACES_PER_SOCKET = 12

if __name__ == "__main__":
  header = f"nodes={NNODES} sockets={NSOCKETS} THIN places={OMP_PLACES} places_bind={OMP_PROC_BIND} places_per_socket={PLACES_PER_SOCKET} size={SIZE} Imax={IMAX} xL={X_L} yL={Y_L} xR={X_R} yR={Y_R}"
  print("*" * len(header))
  print(header)
  print("*" * len(header))

  for node in range(NSOCKETS, 0, -1):
    places = PLACES_PER_SOCKET
    while places > 0:
      start = time.time()
      call(["make", "run"], env={
        "ARGS": f"{SIZE} {SIZE} {X_L} {Y_L} {X_R} {Y_R} {IMAX} test.pgm",
        "np": str(node),
        "OMP_PLACES": OMP_PLACES,
        "OMP_NUM_THREADS": str(places),
        "OMP_PROC_BIND": OMP_PROC_BIND,
        **os.environ
      }, stdout=DEVNULL, stderr=DEVNULL)
      end = time.time()

      elapsed = end - start
      print(f"nodes={node} | places={places} | {elapsed:.3f}")

      places //= 2