#ifndef __LINEARELASTICITY__
#define __LINEARELASTICITY__

#include <petsc.h>
#include <petsc/private/dmdaimpl.h>
#include <iostream>
#include <math.h>
#include <TopOpt.h>
#include <vector>

/*
 Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013
 
 Modified by Oded Amir and Eilam Amir, January 2021

 Disclaimer:
 The authors reserve all rights but do not guarantee that the code is
 free from errors. Furthermore, we shall not be liable in any event
 caused by the use of the program.
*/


class LinearElasticity{

public:
    // Constructor
    LinearElasticity(TopOpt *opt);

    // Destructor
    ~LinearElasticity();

    //  Compute objective and constraints and sensitivities at once: GOOD FOR SELF_ADJOINT PROBLEMS
    PetscErrorCode ComputeObjectiveConstraintsSensitivities(TopOpt *opt);

    // Get pointer to the FE solution
    Vec GetStateField(PetscInt i){ return(U[i]); };
    	
	std::vector<Vec> RHS; // Array of load vectors
	std::vector<Vec> U; // Array of displacement vectors
	std::vector<Vec> N; // Array of BC vectors
	
	int ijkmat[20][20];
	
	// New stuff for staged construction
	std::vector<Vec> RHSstg; // Array of load vectors
	std::vector<Vec> Ustg; // Array of displacement vectors
	Vec Nstg; // BC vector
	
	// Compute additional term of objective function and its sensitivities
	PetscErrorCode ComputeObjectiveSensitivitiesSTG(TopOpt *opt);
	
private:

    PetscScalar KE[24*24]; // Element stiffness matrix 
	PetscScalar BMatrix[6][24]; // B-Matrix for stress computations
    Mat K; // Global stiffness matrix
	Mat Ksolve; // Copy of stiffness matrix for repeated solutions
    	
	Vec NI; // For Dirichlet bc
	
    // Solver
    KSP ksp;	// Pointer to the KSP object i.e. the linear solver+prec

    // Set up the FE mesh and data structures
    PetscErrorCode SetUpLoadAndBC(TopOpt *opt);

    // Solve the FE problem
    PetscErrorCode SolveState(TopOpt *opt);

    // Assemble the stiffness matrix
    PetscErrorCode AssembleStiffnessMatrix(TopOpt *opt);

    // Start the solver
    PetscErrorCode SetUpSolver(TopOpt *opt);

    // Routine that doesn't change the element type upon repeated calls
    PetscErrorCode DMDAGetElements_3D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[]);

    // Methods used to assemble the element stiffness matrix
    PetscInt Hex8Isoparametric(PetscScalar *X, PetscScalar *Y, PetscScalar *Z, PetscScalar nu, PetscInt redInt, PetscScalar *ke, PetscScalar BMatrix[6][24]);
    PetscScalar Dot(PetscScalar *v1, PetscScalar *v2, PetscInt l);
    void DifferentiatedShapeFunctions(PetscScalar xi, PetscScalar eta, PetscScalar zeta, PetscScalar *dNdxi, PetscScalar *dNdeta, PetscScalar *dNdzeta);
    PetscScalar Inverse3M(PetscScalar J[][3], PetscScalar invJ[][3]);

	// New stuff for staged construction
	Mat Kstg; // Global stiffness matrix
	KSP kspSTG; 
	PetscErrorCode SetUpSolverSTG(TopOpt *opt);
	
	// Assemble the stiffness matrix
    PetscErrorCode AssembleStiffnessMatrixSTG(TopOpt *opt, PetscInt slc);
	
	// Assemble the self-weight load vector
	PetscErrorCode AssembleLoadSTG(TopOpt *opt, PetscInt slc);
	
	// Solve the FE problem
    PetscErrorCode SolveStateSTG(TopOpt *opt, PetscInt slc);
	
	PetscInt Hex8Lattice(PetscScalar *X, PetscScalar *Y, PetscScalar *Z, PetscInt redInt, PetscScalar dens, PetscScalar nu0, PetscScalar E0, PetscScalar Emin, PetscScalar *ke);
	
	PetscInt Hex8LatticeDeriv(PetscScalar *X, PetscScalar *Y, PetscScalar *Z, PetscInt redInt, PetscScalar dens, PetscScalar nu0, PetscScalar E0, PetscScalar Emin, PetscScalar *dke);
	
};

#endif
