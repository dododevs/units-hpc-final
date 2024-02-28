CC = mpicc
CFLAGS = -Wall -Wextra -fopenmp
LDFLAGS = -lm
EXEC = mandelbrot
SRCS = mandelbrot.c pgm.c
OBJS = $(SRCS:.c=.o)
MPIRUN_FLAGS = --mca mca_base_component_show_load_errors 0 --map-by core --use-hwthread-cpus

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(EXEC) $(OBJS)

run:
	mpirun $(MPIRUN_FLAGS) -np $(np) $(EXEC) $(ARGS)
