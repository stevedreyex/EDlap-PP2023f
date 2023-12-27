all:
	gcc seq.c -lpng -o seq.o
	gcc omp.c -lpng -fopenmp -o omp.o
	mpicc mpi.c -lpng -o mpi.o

seq:
	./seq.o rem.png

omp:
	./omp.o rem.png 

openmpi:
	sbatch mpi.job

mpi:	
	mpirun -np 10 ./mpi.o rem.png

clean: 
	rm *.o
	rm output_*
	rm slurm*.out

prepare:
	sudo apt-get install libpng-dev
