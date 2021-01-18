#include <petsc.h>
#include <TopOpt.h>
#include <LinearElasticity.h>
#include <MMA.h>
#include <Filter.h>
#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

/*
 Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013
 
 Modified by Oded Amir and Eilam Amir, January 2021

 Disclaimer:
 The authors reserve all rights but do not guarantee that the code is
 free from errors. Furthermore, we shall not be liable in any event
 caused by the use of the program.
*/

// Large-scale Topopt
// Extension for staged construction
// Two-material solution: solid/void and supporting lattice 

static char help[] = "Minimum Weighted Compliance s.t. Vol with Staged Construction \n";

int main(int argc, char *argv[]){

	// Error code for debugging
	PetscErrorCode ierr;

	// Initialize PETSc / MPI and pass input arguments to PETSc
	PetscInitialize(&argc,&argv,PETSC_NULL,help);
  
	// STEP 1: THE OPTIMIZATION PARAMETERS, DATA AND MESH (!!! THE DMDA !!!)
	TopOpt *opt = new TopOpt();

	// STEP 2: THE PHYSICS
	LinearElasticity *physics = new LinearElasticity(opt);

	// STEP 3: THE FILTERING
	Filter *filter = new Filter(opt);

	// STEP 5: THE OPTIMIZER MMA
	MMA *mma;
	PetscInt itr=0;
	opt->AllocMMAwithRestart(&itr, &mma); // allow for restart !

	// STEP 6: FILTER THE INITIAL DESIGN
	// Transfer the complete x to separate x1 and x2
	ierr = VecStrideGather(opt->x,0,opt->x1,INSERT_VALUES);
	ierr = VecStrideGather(opt->x,1,opt->x2,INSERT_VALUES);
	ierr = filter->FilterProject(opt); CHKERRQ(ierr);
	// At this point, xTilde1 and xTilde2 contain filtered values
	
	// Ensure that solid features are maintained in the design x1
	Vec xDoubleTilde;
	Vec OneMxsolid; // For sensitivities
	ierr = VecDuplicate(opt->xTilde1,&(xDoubleTilde)); CHKERRQ(ierr);
	ierr = VecCopy(opt->xTilde1,xDoubleTilde); CHKERRQ(ierr); // xDoubleTilde holds filtered field 
	ierr = VecDuplicate(opt->xsolid1,&(OneMxsolid)); CHKERRQ(ierr);
	ierr = VecCopy(opt->xsolid1,OneMxsolid); CHKERRQ(ierr); // OneMxsolid holds xsolid1
	ierr = VecScale(OneMxsolid,-1.0); // OneMxsolid holds -xsolid1
	ierr = VecShift(OneMxsolid,1.0); // OneMxsolid holds 1-xsolid1
	// Compute xDoubleTilde = xtilde1 + (1-xtilde1).*xsolid
	ierr = VecScale(xDoubleTilde,-1.0); // Compute -xTilde
	ierr = VecShift(xDoubleTilde,1.0); // Compute 1-xTilde
	ierr = VecPointwiseMult(xDoubleTilde,xDoubleTilde,opt->xsolid1); // Multiply by xsolid1
	ierr = VecAXPY(xDoubleTilde,1.0,opt->xTilde1); // Add xtilde1
	
	// For output
	MPI_Comm comm;
	comm = PETSC_COMM_WORLD;
	
	// For previews
	Vec xPhysPreview;
	ierr = VecDuplicate(opt->xPhys1,&(xPhysPreview)); CHKERRQ(ierr);
	
	// For Heaviside projections
	PetscScalar betaHS = 1.0;
	PetscScalar betaHSmax = 8.0;
	opt->penal = 1.0;
	PetscScalar penalmax = 1.0; // No penalty, projections are enough
	PetscScalar etaplus = 0.75;
	PetscScalar etaminus = 0.25;
	ierr = VecCopy(opt->xPhys1,opt->xEro1); CHKERRQ(ierr);
	ierr = VecCopy(opt->xPhys1,opt->xDil1); CHKERRQ(ierr);
	
	PetscScalar *xt2, *xt3, *xp;
	Vec xTildeSolid;
	ierr = VecDuplicate(opt->xPhys1,&(xTildeSolid)); CHKERRQ(ierr);	
	Vec xTildeSupport;
	ierr = VecDuplicate(opt->xPhys2,&(xTildeSupport)); CHKERRQ(ierr);
	Vec xTilde2Mat;
	ierr = VecDuplicate(opt->xPhys1,&(xTilde2Mat)); CHKERRQ(ierr);
	
	//////////////////////////
	// Project design field //
	//////////////////////////
	PetscScalar nom, denomphys, denomerodil, dx;
	PetscScalar *xt1, *xp1, *xero1, *xdil1;
	PetscScalar *df, *dg0, *dg1, *onemxp;
	PetscInt locsiz;
		
	VecGetArray(opt->xPhys1,&xp1);
	VecGetArray(opt->xEro1,&xero1);
	VecGetArray(opt->xDil1,&xdil1);
	VecGetArray(xDoubleTilde,&xt1); // This is now the filtered field1 + solids
	VecGetLocalSize(opt->x1,&locsiz);
	
	denomphys = 2.0*PetscTanhReal(betaHS*0.5);
	denomerodil = PetscTanhReal(betaHS*etaplus) + PetscTanhReal(betaHS*etaminus);
	
	for (PetscInt i=0;i<locsiz;i++){
		// Field x1 - "filtered" field is xDoubleTilde 
		nom = PetscTanhReal(betaHS*0.5) + PetscTanhReal(betaHS*(xt1[i]-0.5));
		xp1[i] = nom/denomphys;
		nom = PetscTanhReal(betaHS*etaplus) + PetscTanhReal(betaHS*(xt1[i]-etaplus));
		xero1[i] = nom/denomerodil;
		nom = PetscTanhReal(betaHS*etaminus) + PetscTanhReal(betaHS*(xt1[i]-etaminus));
		xdil1[i] = nom/denomerodil;	
	}
	
	VecRestoreArray(opt->xPhys1,&xp1);
	VecRestoreArray(opt->xEro1,&xero1);
	VecRestoreArray(opt->xDil1,&xdil1);
	VecRestoreArray(xDoubleTilde,&xt1);
	//////////////////////////
	
	// Set volume fraction
	opt->volfrac1dil = 1.0*opt->volfrac1; 
	
	// For viewing
	PetscViewer viewerVTS;
	
	// STEP 7: OPTIMIZATION LOOP
	PetscScalar ch = 1.0;
	double t1,t2;
  
	while (itr < opt->maxItr && ch > 0.001){
		// Update iteration counter
		itr++;

		// start timer
		t1 = MPI_Wtime();
		
		// // Check bounds
		// {
			// PetscViewer viewer;
			// ierr = PetscViewerVTKOpen(comm,"../output/xsolid.vts",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr); // Open viewer 
			// ierr = VecView(opt->xsolid1,viewer); CHKERRQ(ierr); // View vector 
			// ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr); // Close viewer
			// ierr = PetscViewerVTKOpen(comm,"../output/xvoid.vts",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr); // Open viewer 
			// ierr = VecView(opt->xvoid1,viewer); CHKERRQ(ierr); // View vector 
			// ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr); // Close viewer
		// }
		
		// // Check bounds
		// {
			// PetscViewer viewer;
			// ierr = PetscViewerVTKOpen(comm,"../output/XMIN.vts",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr); // Open viewer 
			// ierr = VecView(opt->XMIN,viewer); CHKERRQ(ierr); // View vector 
			// ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr); // Close viewer
			// ierr = PetscViewerVTKOpen(comm,"../output/XMAX.vts",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr); // Open viewer 
			// ierr = VecView(opt->XMAX,viewer); CHKERRQ(ierr); // View vector 
			// ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr); // Close viewer
		// }

		// Compute obj+const+sens for final compliance
		ierr = physics->ComputeObjectiveConstraintsSensitivities(opt); CHKERRQ(ierr);
		
		// Create reduced design fields according to slicing
		for(PetscInt i=0; i<opt->nslc; i++){
			ierr = VecPointwiseMult(opt->xSTGload[i],opt->xEro1,opt->slcInputload[i]); CHKERRQ(ierr);
			ierr = VecPointwiseMult(opt->xSTGstiff1[i],opt->xEro1,opt->slcInputstiff[i]); CHKERRQ(ierr);
			ierr = VecPointwiseMult(opt->xSTGstiff2[i],opt->xTilde2,opt->slcInputstiff[i]); CHKERRQ(ierr);
		}
		
		// Compute obj+cons+sens for staged compliance
		ierr = physics->ComputeObjectiveSensitivitiesSTG(opt); CHKERRQ(ierr);
		
		// Spit out compliances separately
		PetscPrintf(PETSC_COMM_WORLD,"True compliances are: %f %f \n",opt->fx,opt->compSTG);
		
		// Sum both compliances and derivatives
		opt->fx = opt->fx + opt->printweight*opt->compSTG;
		ierr = VecAXPY(opt->dfdx,opt->printweight,opt->dfdxSTG); CHKERRQ(ierr);
		
		// Compute objective scale
		if (itr==1){
			opt->fscale = 10.0/opt->fx;
		}

		// Scale objective and sensitivities
		opt->fx = opt->fx*opt->fscale;
		VecScale(opt->dfdx,opt->fscale);

		///////////////////////////////
		// Chain rule for projection //
		///////////////////////////////
		VecGetArray(xDoubleTilde,&xt1); // This is now the filtered field1 + solids 
		VecGetArray(opt->dfdx,&df);
		VecGetArray(opt->dgdx[0],&dg0);
		VecGetArray(OneMxsolid,&onemxp);
		VecGetLocalSize(opt->x1,&locsiz);

		denomphys = 2.0*PetscTanhReal(betaHS*0.5);
		denomerodil = PetscTanhReal(betaHS*etaplus) + PetscTanhReal(betaHS*etaminus);
		
		for (PetscInt i=0;i<locsiz;i++){
			// Use xEro for compliance in field x1
			nom = PetscTanhReal(betaHS*(xt1[i]-etaplus));
			nom = betaHS*(1.0 - PetscPowScalar(nom,2.0));
			dx = nom/denomerodil;
			df[2*i] = df[2*i]*dx*onemxp[i];
			// Use xDil for volume in field x1
			nom = PetscTanhReal(betaHS*(xt1[i]-etaminus));
			nom = betaHS*(1.0 - PetscPowScalar(nom,2.0));
			dx = nom/denomerodil;
			dg0[2*i] = dg0[2*i]*dx*onemxp[i];
		}
		VecRestoreArray(xDoubleTilde,&xt1);
		VecRestoreArray(opt->dfdx,&df);
		VecRestoreArray(opt->dgdx[0],&dg0);
		VecRestoreArray(OneMxsolid,&onemxp);
		///////////////////////////////
		
		// // Here we need to chain xDoubleTilde to xTilde 
		// THIS CHAIN RULE IS NOT INCLUDED IN THE CURRENT IMPLEMENTATION - YET TO BE VERIFIED!!!
		// ierr = VecPointwiseMult(opt->dfdx,opt->dfdx,OneMxsolid); // Multiply dfdx with OneMxsolid
		// ierr = VecPointwiseMult(opt->dgdx[0],opt->dgdx[0],OneMxsolid); // Multiply dfdx with OneMxsolid
		
		// Filter sensitivities (chain  rule) 
		// Only PDE filter is implemented so far!!!
		ierr = filter->Gradients(opt); CHKERRQ(ierr);
		
		// // Check sensitivities
		// {
			// PetscViewer viewer;
			// ierr = PetscViewerVTKOpen(comm,"../output/dfdx.vts",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr); // Open viewer 
			// ierr = VecView(opt->dfdx,viewer); CHKERRQ(ierr); // View vector 
			// ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr); // Close viewer
			// ierr = PetscViewerVTKOpen(comm,"../output/dg0dx.vts",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr); // Open viewer 
			// ierr = VecView(opt->dgdx[0],viewer); CHKERRQ(ierr); // View vector 
			// ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr); // Close viewer
			// ierr = PetscViewerVTKOpen(comm,"../output/dg1dx.vts",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr); // Open viewer 
			// ierr = VecView(opt->dgdx[1],viewer); CHKERRQ(ierr); // View vector 
			// ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr); // Close viewer
		// }
		
		// Set outer move limits on design variables, box constraints are vectors
		ierr = mma->SetOuterMovelimitVec(opt->XMIN,opt->XMAX,opt->movlim,opt->x,opt->xmin,opt->xmax); CHKERRQ(ierr);
		
		// Update design by MMA
		ierr = mma->Update(opt->x,opt->dfdx,opt->gx,opt->dgdx,opt->xmin,opt->xmax); CHKERRQ(ierr);

		// Inf norm on the design change
		ch = mma->DesignChange(opt->x,opt->xold);

		// Filter design field
		ierr = VecStrideGather(opt->x,0,opt->x1,INSERT_VALUES);
		ierr = VecStrideGather(opt->x,1,opt->x2,INSERT_VALUES);
		ierr = filter->FilterProject(opt); CHKERRQ(ierr);
		// At this point, xtilde1 and xtilde2 contain filtered values
		
		// Ensure that solid features are maintained in the design
		ierr = VecCopy(opt->xTilde1,xDoubleTilde); CHKERRQ(ierr); // xDoubleTilde holds filtered field 
		// Compute xDoubleTilde = xtilde1 + (1-xtilde1).*xsolid
		ierr = VecScale(xDoubleTilde,-1.0); // Compute -xTilde
		ierr = VecShift(xDoubleTilde,1.0); // Compute 1-xTilde
		ierr = VecPointwiseMult(xDoubleTilde,xDoubleTilde,opt->xsolid1); // Multiply by xsolid1
		ierr = VecAXPY(xDoubleTilde,1.0,opt->xTilde1); // Add xtilde1

		//////////////////////////
		// Project design field //
		//////////////////////////
		// Update betaHS
		PetscInt pace = opt->maxItr/4;
		if (itr%pace==0 || ch < 0.002 ){
			opt->penal = PetscMin(opt->penal+0.5,penalmax);
			PetscPrintf(PETSC_COMM_WORLD,"Penalty raised to: %f\n",opt->penal);
			if (opt->penal > 0.5){
				betaHS = PetscMin(betaHS*2.0,betaHSmax);
				PetscPrintf(PETSC_COMM_WORLD,"betaHS raised to: %f\n",betaHS);
			}
			opt->printweight = PetscMax(opt->printweight/1.0,1.0e-10);
			PetscPrintf(PETSC_COMM_WORLD,"PrintWeight changed to: %f\n",opt->printweight);
		}

		VecGetArray(opt->xPhys1,&xp1);
		VecGetArray(opt->xEro1,&xero1);
		VecGetArray(opt->xDil1,&xdil1);
		VecGetArray(xDoubleTilde,&xt1); // This is now the filtered field1 + solids
		VecGetLocalSize(opt->x1,&locsiz);
		
		denomphys = 2.0*PetscTanhReal(betaHS*0.5);
		denomerodil = PetscTanhReal(betaHS*etaplus) + PetscTanhReal(betaHS*etaminus);
		
		for (PetscInt i=0;i<locsiz;i++){
			// Field x1 - "filtered" field is xDoubleTilde 
			nom = PetscTanhReal(betaHS*0.5) + PetscTanhReal(betaHS*(xt1[i]-0.5));
			xp1[i] = nom/denomphys;
			nom = PetscTanhReal(betaHS*etaplus) + PetscTanhReal(betaHS*(xt1[i]-etaplus));
			xero1[i] = nom/denomerodil;
			nom = PetscTanhReal(betaHS*etaminus) + PetscTanhReal(betaHS*(xt1[i]-etaminus));
			xdil1[i] = nom/denomerodil;	
		}
		
		VecRestoreArray(opt->xPhys1,&xp1);
		VecRestoreArray(opt->xEro1,&xero1);
		VecRestoreArray(opt->xDil1,&xdil1);
		VecRestoreArray(xDoubleTilde,&xt1);
		//////////////////////////

		// Stop timer
		t2 = MPI_Wtime();

		// Print to screen
		PetscPrintf(PETSC_COMM_WORLD,"It.: %i, obj.: %f, g[0]: %f, g[1]: %f, ch.: %f, time: %f\n",
			itr,opt->fx/opt->fscale,opt->gx[0],opt->gx[1],ch,t2-t1);

		if (itr%5==0) { // Spit out every 5 iters starting at 5
			// Output xTilde to vts
			char xphysfilename [PETSC_MAX_PATH_LEN];
			int nfilename;
			nfilename = sprintf (xphysfilename,"../output/xPhys1_preview.vts");
			ierr = PetscViewerVTKOpen(comm,xphysfilename,FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer 
			ierr = VecView(opt->xTilde1,viewerVTS); CHKERRQ(ierr); 
			PetscViewerDestroy(&viewerVTS);
			nfilename = sprintf (xphysfilename,"../output/xPhys2_preview.vts");
			ierr = PetscViewerVTKOpen(comm,xphysfilename,FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer 
			ierr = VecView(opt->xTilde2,viewerVTS); CHKERRQ(ierr); 
			PetscViewerDestroy(&viewerVTS);
		}
	
		if (itr%5==0) {
			// Adjust volume fraction for x1
			PetscScalar sumxPhys;
			ierr = VecSum(opt->xPhys1,&sumxPhys); CHKERRQ(ierr);
			PetscReal ratio = opt->volfrac1/sumxPhys*((PetscScalar)opt->nvoxel);
			opt->volfrac1dil = opt->volfrac1dil*PetscPowScalar(ratio,0.5);
		}

	} // End of optimization iterations
	
	VecDestroy(&xDoubleTilde);
	VecDestroy(&OneMxsolid);
	VecDestroy(&xPhysPreview);
	
	// STEP 7: CLEAN UP AFTER YOURSELF
	delete mma;
	delete filter;
	delete physics;
	delete opt;
	
	// Finalize PETSc / MPI
	PetscFinalize();
	return 0;
}
