#ifndef PDE_FILTER_H
#define PDE_FILTER_H
#include <TopOpt.h>
#include <petsc.h>
#include <petsc/private/dmdaimpl.h>

class PDEFilt
{


	public:

	PDEFilt(TopOpt *opt);
	~PDEFilt();

	PetscErrorCode FilterProject(Vec XX, Vec F);
	PetscErrorCode Gradients(Vec OS, Vec FS);

	private:

	
	PetscInt nn[3]; // Number of nodes in each direction
	PetscInt ne[3]; // Number of elements in each direction
	PetscScalar xc[6]; // Domain coordinates
        PetscScalar elemVol; // element volume

	PetscScalar R; //filter parameter

	PetscScalar KF[8*8]; // PDE filter stiffness matrix
        PetscScalar TF[8];   // PDE filter transformation matrix


        PetscInt nloc; // Number of local nodes?

	PetscInt nlvls; // Number of multigrid levels for the filter
	
	DM da_nodal; 
	DM da_element;

	Mat K; // Global stiffness matrix
	Mat T; // Transformation matrix   RHS=T*X 
	Vec RHS; // Load vector - nodal
	Vec U;
	Vec X;  //filtered filed - element

	KSP ksp; // linear solver

	void PDEFilterMatrix(PetscScalar dx, PetscScalar dy, PetscScalar dz, 
							   PetscScalar R, 
							PetscScalar *KK, PetscScalar *T);

	PetscErrorCode DMDAGetElements_3D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[]);

	void MatAssemble(); //assemble K and T
			    // RHS = T*elvol*RHO

	PetscErrorCode SetUpSolver();
	PetscErrorCode Free();


};






#endif
