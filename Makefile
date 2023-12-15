all:
	gcc seq.c -lpng -o seq.o
	gcc omp.c -lpng -fopenmp -o omp.o
	mpicc mpi.c -lpng -o mpi.o
	nvcc -O3 host.cu kernel.cu -lpng -g -o cuda.o

seq:
	./seq.o test-image.png

omp:
	./omp.o test-image.png 

openmpi:
	sbatch mpi.job

mpi:	
	mpirun -np 10 ./mpi.o test-image.png

cuda:
	./cuda.o test-image.png

clean: 
	rm *.o
	rm output_*
	rm slurm*.out

prepare:
	sudo apt-get install libpng-dev
