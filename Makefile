CXX = g++
MPIXX = mpicxx
CXXFLAGS = -std=c++11 -O2

all: blocking nonblocking

blocking: life-blocking.C
	$(MPIXX) $(CXXFLAGS) -o life-blocking $<

nonblocking: life-nonblocking.C
	$(MPIXX) $(CXXFLAGS) -o life-nonblocking $<

clean:
	rm -f life-blocking life-nonblocking 
