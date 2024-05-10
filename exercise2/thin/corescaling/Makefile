CC = mpicc
CFLAGS = -Wall -Wextra -fopenmp
LDFLAGS = -lm
LDFLAGS_VIZ = -lm -lSDL2
EXEC = mandelbrot
SRCS = mandelbrot.c pgm.c log.c viz.c
OBJS = $(SRCS:.c=.o)
MPIRUN_FLAGS = --mca mca_base_component_show_load_errors 0 --map-by core

LOG_LEVEL = 1

all:
	$(CC) $(CFLAGS) -DLOG_LEVEL=$(LOG_LEVEL) -o $(EXEC) $(SRCS) $(LDFLAGS)

viz:
	$(CC) $(CFLAGS) -DLOG_LEVEL=$(LOG_LEVEL) -D VIZ -o $(EXEC) $(SRCS) $(LDFLAGS_VIZ)

clean:
	rm -f $(EXEC) $(OBJS)

run:
	mpirun $(MPIRUN_FLAGS) -np $(np) $(EXEC) $(ARGS)
