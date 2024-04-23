from subprocess import call, DEVNULL
import time
import os

SIZE = 1024
IMAX = 65535
X_L = -3
Y_L = -3
X_R = 3
Y_R = 3

MIN_PROCESSES = 2
MAX_PROCESSES = 16

if __name__ == "__main__":
  header = f"size = {SIZE} Imax = {IMAX} xL = {X_L} yL = {Y_L} xR = {X_R} yR = {Y_R}"
  print("*" * len(header))
  print(header)
  print("*" * len(header))

  for np in range(MAX_PROCESSES, MIN_PROCESSES - 1, -1):
    start = time.time()
    call(["make", "run"], env={
      "ARGS": f"{SIZE} {SIZE} {X_L} {Y_L} {X_R} {Y_R} {IMAX} test.pgm",
      "np": str(np),
      **os.environ
    }, stdout=DEVNULL, stderr=DEVNULL)
    end = time.time()

    elapsed = end - start
    print(f"np={np} | {elapsed:.3f}")
