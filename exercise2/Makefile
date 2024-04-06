CC = mpicc
CFLAGS = -Wall -Wextra -fopenmp
LDFLAGS = -lm
LDFLAGS_VIZ = -lm -lSDL2
EXEC = mandelbrot
SRCS = mandelbrot.c pgm.c log.c viz.c
OBJS = $(SRCS:.c=.o)
MPIRUN_FLAGS = --mca mca_base_component_show_load_errors 0 --map-by core --use-hwthread-cpus

all:
	$(CC) $(CFLAGS) -o $(EXEC) $(SRCS) $(LDFLAGS)

viz:
	$(CC) $(CFLAGS) -D VIZ -o $(EXEC) $(SRCS) $(LDFLAGS_VIZ)

clean:
	rm -f $(EXEC) $(OBJS)

run:
	mpirun $(MPIRUN_FLAGS) -np $(np) $(EXEC) $(ARGS)
