#ifndef TOPOPT_H
#define TOPOPT_H

#include <petsc.h>
#include <petscdmforest.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <MMA.h>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <vector>

class TopOpt {

public:

	// Constructor/Destructor
	TopOpt();
	~TopOpt();

	// Method to allocate MMA with/without restarting
	void AllocMMAwithRestart(int *itr, MMA **mma);
	void WriteRestartFiles(int *itr, MMA *mma);

	// Physical domain variables
	PetscScalar xc[6]; // Domain coordinates
	PetscScalar dx,dy,dz; // Element size
	PetscInt nxyz[3]; // Number of nodes in each direction
	PetscInt nlvls; // Number of multigrid levels
	PetscScalar nu; // Poisson's ratio

	// Nodal mesh (contains physics)
	DM da_nodes;
	// element mesh (contains design)
	DM da_elem;
	DM da_elem2;

	// Optimization parameters
	PetscInt n; // Total number of design variables
	PetscInt nvoxel; // Total number of voxels
	PetscInt nloc; // Local number of local nodes
	PetscInt m; // Number of constraints
	PetscScalar fx; // Objective value
	PetscScalar fscale; // Scaling factor for objective
	PetscScalar *gx; // Array with constraint values
	PetscScalar Xmin; // Min. value of design variables
	PetscScalar Xmax; // Max. value of design variables

	PetscScalar movlim; // Max. change of design variables
	PetscScalar volfrac1; // Volume fraction for solid material
	PetscScalar volfrac1dil; // Volume fraction for solid material on dilated design
	PetscScalar volfrac2; // Volume fraction for lattice material 
	PetscScalar penal; // Penalization parameter
	PetscScalar Emin, Emax; // Modified SIMP, max and min E
	PetscScalar Sy, density; // Yield stress, material density

	PetscScalar rmin; // filter radius

	PetscInt maxItr; // Max iterations
	PetscInt filter; // Filter type

	PetscInt nlc[8]; // Number of load cases per BC (max 8 BCs)
	PetscInt nbc; // Number of BCs
	PetscInt tnlc; // Total number of load cases
	PetscScalar *weights; // Weights for minimum weighted compliance
	
	Vec x; // Design variables - all (for MMA basically)
	Vec x1; // Design variables - solid/void
	Vec x2; // Design variables - lattice 
	Vec xTilde1; // Filtered variables - solid/void
	Vec xTilde2; // Filtered variables - lattice 
	Vec xPhys1; // Physical variables 
	Vec xPhys2; // Physical variables 
	Vec xDil1; // Dilated densities
	Vec xEro1; // Eroded densities
	Vec xvoid1; // Points to void elements
    Vec xsolid1; // Points to solid elements
	
	Vec dfdx; // Sensitivities of objective
	Vec xmin, xmax; // Vectors with max and min values of x
	Vec xold; // x from previous iteration
	Vec *dgdx; // Sensitivities of constraints (vector array)

    Vec XMIN; // Lower bound considering solids and voids
    Vec XMAX; // Upper bound considering solids and voids

	// Restart data for MMA:
	PetscBool restart, flip;
	std::string restdens_1,restdens_2;
	Vec xo1, xo2, U, L;
	
	PetscReal volfracin1, volfracin2;
	
	// New stuff for staged construction
	std::vector<Vec> xSTGload; // Array of densities for stages (size nslc)
	std::vector<Vec> xSTGstiff1; // Array of densities for stages (size nslc)
	std::vector<Vec> xSTGstiff2; // Array of densities for stages (size nslc)
	std::vector<Vec> slcInputload; // Array of 0-1 vectors defining slices (size nslc)
	std::vector<Vec> slcInputstiff; // Array of 0-1 vectors defining slices (size nslc)
	
	PetscInt nslc; // Number of slices 
	PetscInt interprule; // Interpolation rule (1==SIMP, 2==LATTICE)
	PetscInt printdir; // Printing direction (0==x, 1==y, 2==z)
	PetscScalar printweight; // Relative weight of staged construction
	PetscScalar compSTG; // Compliance of staged construction
	Vec dfdxSTG; // Sensitivities of staged construction

private:
	// Allocate and set default values
	PetscErrorCode SetUp();
	PetscErrorCode SetUpMESH();
	PetscErrorCode SetUpOPT();

};

#endif
