
.PHONY: all clean test

all:
	env F90FLAGS=-fopenmp python setup.py build_ext --inplace --fcompiler=intel

test: all
	@echo Running with 1 thread...
	env OMP_NUM_THREADS=1 python script.py
	@echo Running with 2 thread...
	env OMP_NUM_THREADS=2 python script.py
	@echo Running with 4 thread...
	env OMP_NUM_THREADS=4 python script.py
	@echo Running with 8 thread...
	env OMP_NUM_THREADS=8 python script.py

clean:
	rm -rf build
	rm -rf wrapper.so
