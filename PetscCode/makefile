PETSC_DIR=/u/eilam.amir/soft/petsc-3.7.7
PETSC_ARCH=eilam-debug
CFLAGS = -I.
FFLAGS=
CPPFLAGS=-I.
FPPFLAGS=
LOCDIR=
EXAMPLESC=
EXAMPLESF=
MANSEC=
CLEANFILES=
NP=

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

topopt: main.o TopOpt.o LinearElasticity.o MMA.o Filter.o PDEFilter.o chkopts
	rm -rf topopt
	-${CLINKER} -o topopt main.o TopOpt.o LinearElasticity.o MMA.o Filter.o PDEFilter.o ${PETSC_SYS_LIB}
	${RM}  main.o TopOpt.o LinearElasticity.o MMA.o Filter.o PDEFilter.o
	rm -rf *.o

myclean:
	rm -rf minwc *.o output* binary* log* restore* 