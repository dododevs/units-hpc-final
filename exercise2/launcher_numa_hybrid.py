from subprocess import call, DEVNULL, PIPE, Popen
import time
import os
import re

SIZE = 512
IMAX = 65535
X_L = -3
Y_L = -3
X_R = 3
Y_R = 3

# MIN_PROCESSES = 1
# MAX_PROCESSES = 2

# MIN_PLACES = 1
# MAX_PLACES = 128

def get_numa_count():
  p = Popen("numactl --hardware", shell=True, stdout=PIPE)
  out, err = p.communicate()
  ptn = re.findall("available: ([0-9]+) nodes", str(out))
  if not ptn:
    return 1
  return int(ptn[0])

def get_omp_num_threads():
  p = Popen("./get_omp_num_threads", env=os.environ, shell=True, stdout=PIPE)
  out, err = p.communicate()
  ptn = re.findall("omp_get_num_threads\(\) = ([0-9]+)", str(out))
  if not ptn:
    return 1
  return int(ptn[0])

if __name__ == "__main__":
  header = f"size = {SIZE} Imax = {IMAX} xL = {X_L} yL = {Y_L} xR = {X_R} yR = {Y_R}"
  print("*" * len(header))
  print(header)
  print("*" * len(header))

  nnuma = get_numa_count()
  nthreads = get_omp_num_threads()

  for numa in range(nnuma, 0, -1):
    for thread in range(nthreads, 0, -1):
      start = time.time()
      call(["make", "run"], env={
        "ARGS": f"{SIZE} {SIZE} {X_L} {Y_L} {X_R} {Y_R} {IMAX} test.pgm",
        "np": str(numa),
        "OMP_PLACES": "cores",
        "OMP_NUM_THREADS": str(thread),
        **os.environ
      }, stdout=DEVNULL, stderr=DEVNULL)
      end = time.time()

      elapsed = end - start
      print(f"nnuma={numa} | {elapsed:.3f}")
