PS_PATH = -I ../../FDPS/src/
CC = g++
CFLAGS = -O3 -ffast-math -funroll-loops -std=c++11
CFLAGS += -DPARTICLE_SIMULATOR_THREAD_PARALLEL -fopenmp
#CFLAGS += -DPARTICLE_SIMULATOR_MPI_PARALLEL
#CFLAGS += -Wall -Wformat=2 -Wcast-qual -Wcast-align -Wwrite-strings -Wconversion -Wfloat-equal -Wpointer-arith
CFLAGS += -DPARTICLE_SIMULATOR_ALL_64BIT_PRECISION
CFLAGS += -DFAST_COMM_FOR_2ND_EXCHANGE
CFLAGS += -DPARTICLE_SIMULATOR_TWO_DIMENSION

CPPOBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
#CPPOBJS = main.cpp
CPPHDRS = $(wildcard *.h)
PROGRAM = sph.out

.PHONY:	clean all

all:	$(CPPOBJS) $(CPPHDRS)
	@echo "Linking object files..."
	@$(CC) $(CFLAGS) $(WARNINGS) $(CPPOBJS) -o $(PROGRAM) $(LIBS) $(PS_PATH)
	@echo "Link Success! [$(PROGRAM)]"

%.o:	%.cpp $(CPPHDRS)
	@echo "Building $< ..."
	@$(CC) -c $< $(CFLAGS) $(WARNINGS) $(PS_PATH)
	@echo "[$< OK]"

clean:
	-rm *.out *.o
