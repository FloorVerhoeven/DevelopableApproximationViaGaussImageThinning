CC = g++
CCFLAGS = -std=c++11 -O3 -Wno-format
INC_DIRS = /usr/local/include /usr/local/include/eigen3 
INC_FLAGS = $(addprefix -I,$(INC_DIRS))
EXAMPLES = architecture boat coffee cone cylinder  dome drill face fandisk  guitar lilium mask nut swing dog fertility bumpy bunny cone_high

GaussThinningParallel: mainParallel.cpp
	$(CC) $(CCFLAGS) -fopenmp $(INC_FLAGS) mainParallel.cpp -o GaussThinningParallel

GaussThinning: main.cpp
	 $(CC) $(CCFLAGS) $(INC_FLAGS) main.cpp -o GaussThinning

clean:
	rm GaussThinning
