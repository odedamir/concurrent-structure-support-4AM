#ifndef __FILTER__
#define __FILTER__

#include <petsc.h>
#include <petsc/private/dmdaimpl.h>
#include <iostream>
#include <math.h>
#include <PDEFilter.h>
#include <TopOpt.h>

class Filter{
  
public:
    // Constructors
    Filter(TopOpt *opt); 
    	
    // Destructors
    ~Filter(); 
	    
    // Filter design variables
    PetscErrorCode FilterProject(TopOpt *opt);
	    
    // Filter the sensitivities
    PetscErrorCode Gradients(TopOpt *opt);
    
private:
  
    // Standard density/sensitivity filter matrix
    Mat H; 		// Filter matrix
    Vec Hs; 		// Filter "sum weight" (normalization factor) vector   
    
    // Mesh used for standard filtering
    DM da_elem;  	// da for image-filter field mesh
    
    // PDE filtering
    PDEFilt *pdef;	// PDE filter class
    
    // Setup datastructures for the filter
    PetscErrorCode SetUp(TopOpt *opt);
	    
    // Routine that doesn't change the element type upon repeated calls
    PetscErrorCode DMDAGetElements_3D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[]);
    
};

#endif
