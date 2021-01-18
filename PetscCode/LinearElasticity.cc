#include <LinearElasticity.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

static int RHSfilter(const struct dirent* dir_ent)
{
    if (!strcmp(dir_ent->d_name, ".") || !strcmp(dir_ent->d_name, "..")) return 0;
    std::string fname = dir_ent->d_name;
    if (fname.find("RHS") == std::string::npos) return 0;
    return 1;
}

LinearElasticity::LinearElasticity(TopOpt *opt){
	
	// Set pointers to null
	K = NULL;
	Ksolve = NULL;
	NI = NULL;
	ksp = NULL;
	
	Kstg = NULL;
	kspSTG = NULL;
	
	// Setup stiffness matrix, load vector and bcs (Dirichlet) for the design problem
	SetUpLoadAndBC(opt);

}

LinearElasticity::~LinearElasticity(){
	
	MatDestroy(&(K));
	MatDestroy(&(Ksolve));
	KSPDestroy(&(ksp));
	
	MatDestroy(&(Kstg));
	KSPDestroy(&(kspSTG));
	
	for(PetscInt i=0; i<U.size(); i++){
		VecDestroy(&(U[i]));
		VecDestroy(&(RHS[i]));
	}

	for(PetscInt i=0; i<N.size(); i++){
		VecDestroy(&(N[i]));
	}
	
	for(PetscInt i=0; i<Ustg.size(); i++){
		VecDestroy(&(Ustg[i]));
		VecDestroy(&(RHSstg[i]));
	}

	VecDestroy(&(Nstg));
	
}

PetscErrorCode LinearElasticity::SetUpLoadAndBC(TopOpt *opt){

	PetscErrorCode ierr;

	// Allocate matrix and the RHS and Solution vector and Dirichlet vector
	ierr = DMCreateMatrix(opt->da_nodes,&(K)); CHKERRQ(ierr);
	ierr = MatDuplicate(K,MAT_SHARE_NONZERO_PATTERN,&(Ksolve)); CHKERRQ(ierr);
	ierr = MatDuplicate(K,MAT_SHARE_NONZERO_PATTERN,&(Kstg)); CHKERRQ(ierr);
	
	// Set U and RHS (loop over total number of loadcases)
	U.resize(opt->tnlc);
	RHS.resize(opt->tnlc);
	for(PetscInt i=0; i<opt->tnlc; i++){
		ierr = DMCreateGlobalVector(opt->da_nodes,&(U[i])); CHKERRQ(ierr);
		VecSet(U[i],0.0);
		ierr = DMCreateGlobalVector(opt->da_nodes,&(RHS[i])); CHKERRQ(ierr);
		VecSet(RHS[i],0.0);
	}
	
	// Set BCs via N vector (loop over total number of BCs)
	N.resize(opt->nbc);	
	for(PetscInt i=0; i<opt->nbc; i++){
		ierr = DMCreateGlobalVector(opt->da_nodes,&(N[i])); CHKERRQ(ierr);
		VecSet(N[i],1.0);
	}
		
	// Set the local stiffness matrix
	PetscScalar a = opt->dx; // x-side length
	PetscScalar b = opt->dy; // y-side length
	PetscScalar c = opt->dz; // z-side length
	PetscScalar X[8] = {0.0, a, a, 0.0, 0.0, a, a, 0.0};
	PetscScalar Y[8] = {0.0, 0.0, b, b, 0.0, 0.0, b, b};
	PetscScalar Z[8] = {0.0, 0.0, 0.0, 0.0, c, c, c, c};

	// Compute the element stiffness matrix - constant due to structured grid
	Hex8Isoparametric(X, Y, Z, opt->nu, false, KE, BMatrix);
	
	//////////////////////////////////////////////////////////////////
	// !!! NO LONGER HOLDS FOR STAGED CONSTRUCTION WITH LATTICE !!! //
	//////////////////////////////////////////////////////////////////
	
	// Read in RHS and N vectors, one by one
    Vec inputvec;
    MPI_Comm comm;
    PetscViewer viewer1;
    comm = PETSC_COMM_WORLD;
	char inputfilename [PETSC_MAX_PATH_LEN];
	int nfilename;
	int lcnum = 0;
	int bcnum = 0;
	int rhsnum = 0;
	
	// Find all RHS*.bin files in input, organize i_j_k indices
	struct dirent **namelist;
    int nfiles;
	
	nfiles = scandir("../input", &namelist, *RHSfilter, alphasort);
    if (nfiles < 0)
        perror("scandir");
    else {
        while (nfiles--) {
      		std::string filestr(namelist[nfiles]->d_name);
			std::string delim="_";
			std::vector<std::string> rez = split(filestr,delim);
			bcnum = atoi(rez[1].c_str());
			rhsnum = atoi(rez[2].c_str());
			ijkmat[bcnum-1][rhsnum-1] = atoi(rez[3].c_str());
			PetscPrintf(PETSC_COMM_WORLD,"Found BC %i and RHS %i and LC %i \n",bcnum,rhsnum,ijkmat[bcnum-1][rhsnum-1]);
            free(namelist[nfiles]);
        }
        free(namelist);
    }
	
	ierr = DMDACreateNaturalVector(opt->da_nodes,&inputvec); CHKERRQ(ierr); CHKERRQ(ierr); // Read in natural ordering
	
	for (int i=0; i<opt->nbc; i++){
		// Read BCs one by one
		nfilename = sprintf(inputfilename,"../input/BC_%i.bin",i+1);
		ierr = PetscViewerBinaryOpen(comm,inputfilename,FILE_MODE_READ,&viewer1); CHKERRQ(ierr); // Read binary file with BC data
		ierr = VecLoad(inputvec,viewer1); CHKERRQ(ierr); // Keep BCs data in new natural vector
		ierr = DMDANaturalToGlobalBegin(opt->da_nodes,inputvec,INSERT_VALUES,N[i]); CHKERRQ(ierr); // Transfer vector to global ordering - start
		ierr = DMDANaturalToGlobalEnd(opt->da_nodes,inputvec,INSERT_VALUES,N[i]); CHKERRQ(ierr); // Transfer vector to global ordering - end
		VecSet(inputvec,0.0);
		
		for (int j=0; j<opt->nlc[i]; j++){
			lcnum = ijkmat[i][j];
			// Read RHS one by one for this BC
			nfilename = sprintf(inputfilename,"../input/RHS_%i_%i_%i.bin",i+1,j+1,lcnum);
			ierr = PetscViewerBinaryOpen(comm,inputfilename,FILE_MODE_READ,&viewer1); CHKERRQ(ierr); // Read binary file with RHS data
			ierr = VecLoad(inputvec,viewer1); CHKERRQ(ierr); // Keep RHS data in new natural vector
			ierr = DMDANaturalToGlobalBegin(opt->da_nodes,inputvec,INSERT_VALUES,RHS[lcnum-1]); CHKERRQ(ierr); // Transfer vector to global ordering - start
			ierr = DMDANaturalToGlobalEnd(opt->da_nodes,inputvec,INSERT_VALUES,RHS[lcnum-1]); CHKERRQ(ierr); // Transfer vector to global ordering - end
			VecSet(inputvec,0.0);
		}
	}
	
	/////////////////////////////////
	// *** STAGED CONSTRUCTION *** //
	/////////////////////////////////
	
	// Set U and RHS (loop over total number of slices)
	Ustg.resize(opt->nslc);
	RHSstg.resize(opt->nslc);
	for(PetscInt i=0; i<opt->nslc; i++){
		ierr = DMCreateGlobalVector(opt->da_nodes,&(Ustg[i])); CHKERRQ(ierr);
		VecSet(Ustg[i],0.0);
		ierr = DMCreateGlobalVector(opt->da_nodes,&(RHSstg[i])); CHKERRQ(ierr);
		VecSet(RHSstg[i],0.0);
	}
	
	// Set BCs via N vector (single vector for building plate)
	ierr = DMCreateGlobalVector(opt->da_nodes,&(Nstg)); CHKERRQ(ierr);
	VecSet(Nstg,1.0);
	ierr = PetscViewerBinaryOpen(comm,"../input/BCstg.bin",FILE_MODE_READ,&viewer1); CHKERRQ(ierr); // Read binary file with BC data
	ierr = VecLoad(inputvec,viewer1); CHKERRQ(ierr); // Keep BCs data in new natural vector
	ierr = DMDANaturalToGlobalBegin(opt->da_nodes,inputvec,INSERT_VALUES,Nstg); CHKERRQ(ierr); // Transfer vector to global ordering - start
	ierr = DMDANaturalToGlobalEnd(opt->da_nodes,inputvec,INSERT_VALUES,Nstg); CHKERRQ(ierr); // Transfer vector to global ordering - end
	VecSet(inputvec,0.0);
	
	// {
		// PetscViewer viewerVTS;
		// ierr = PetscViewerVTKOpen(comm,"../output/nstg_check.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		// ierr = VecView(Nstg,viewerVTS); CHKERRQ(ierr); // View vector
		// ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
	// }
	
	
	VecDestroy(&inputvec);
	ierr = PetscViewerDestroy(&viewer1); CHKERRQ(ierr); // Destroy viewer
	
	/* // View BCs and RHS - for debugging
	{
		PetscViewer viewerVTS;
		ierr = PetscViewerVTKOpen(comm,"../output/RHS1.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		ierr = VecView(RHS[0],viewerVTS); CHKERRQ(ierr); // View vector
		ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
		
		ierr = PetscViewerVTKOpen(comm,"../output/RHS2.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		ierr = VecView(RHS[1],viewerVTS); CHKERRQ(ierr); // View vector
		ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
		
		ierr = PetscViewerVTKOpen(comm,"../output/RHS3.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		ierr = VecView(RHS[2],viewerVTS); CHKERRQ(ierr); // View vector
		ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
		
		ierr = PetscViewerVTKOpen(comm,"../output/RHS4.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		ierr = VecView(RHS[3],viewerVTS); CHKERRQ(ierr); // View vector
		ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
		
		ierr = PetscViewerVTKOpen(comm,"../output/BC1.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		ierr = VecView(N[0],viewerVTS); CHKERRQ(ierr); // View vector
		ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
		
		ierr = PetscViewerVTKOpen(comm,"../output/BC2.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		ierr = VecView(N[1],viewerVTS); CHKERRQ(ierr); // View vector
		ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
	} */
    
	return ierr;
}

PetscErrorCode LinearElasticity::SolveState(TopOpt *opt){

	PetscErrorCode ierr;

	double t1, t2;
	t1 = MPI_Wtime();
	
	// Assemble the stiffness matrix
	ierr = AssembleStiffnessMatrix(opt);
	CHKERRQ(ierr);

	// Solve loadcases
    PetscInt niter;
	PetscScalar rnorm;
	PetscInt lcnum;
	
	////////////////////////////////////
	// Solve a set of RHS for each BC //
	////////////////////////////////////
	VecDuplicate(N[0],&NI);
	for (int i=0; i<opt->nbc; i++){
		// Copy stiffness matrix, keep native K
		MatCopy(K,Ksolve,SAME_NONZERO_PATTERN);
		// Impose the dirichlet conditions, i.e. K = N'*K*N - (N-I)
		// 1.: K = N'*K*N
		// 2. Add ones, i.e. K = K + NI, NI = I - N
		MatDiagonalScale(Ksolve,N[i],N[i]);
		VecSet(NI,1.0);
		VecAXPY(NI,-1.0,N[i]);
		MatDiagonalSet(Ksolve,NI,ADD_VALUES);
		// Setup the solver
		if (ksp==NULL){
			ierr = SetUpSolver(opt);
			CHKERRQ(ierr);
		}
		ierr = KSPSetOperators(ksp,Ksolve,Ksolve);
		CHKERRQ(ierr);
		KSPSetUp(ksp);
		for (int j=0; j<opt->nlc[i]; j++){
			lcnum = ijkmat[i][j];
			// Zero out possible loads in the RHS that coincide with Dirichlet conditions
			VecPointwiseMult(RHS[lcnum-1],RHS[lcnum-1],N[i]);
			// Solve
			ierr = KSPSolve(ksp,RHS[lcnum-1],U[lcnum-1]); CHKERRQ(ierr);
			// Get iteration number and residual from KSP
			KSPGetIterationNumber(ksp,&niter);
			KSPGetResidualNorm(ksp,&rnorm);
			t2 = MPI_Wtime();
			PetscPrintf(PETSC_COMM_WORLD,"State solver, BC %i RHS %i LC %i:  iter: %i, rerr.: %e, time: %f\n",i+1,j+1,lcnum,niter,rnorm,t2-t1);
			t1 = t2;
		}
	}
	
	VecDestroy(&NI);
	
	return ierr;
}

PetscErrorCode LinearElasticity::SolveStateSTG(TopOpt *opt, PetscInt slc){

	PetscErrorCode ierr;

	double t1, t2;
	t1 = MPI_Wtime();
	
	// Assemble the stiffness matrix
	ierr = AssembleStiffnessMatrixSTG(opt,slc);
	CHKERRQ(ierr);

	// Solve loadcases
    PetscInt niter;
	PetscScalar rnorm;
		
	////////////////////////////////
	// Solve a RHS for each slice //
	////////////////////////////////
	
	// Impose the dirichlet conditions, i.e. K = N'*K*N - (N-I)
	// 1.: K = N'*K*N
	// 2. Add ones, i.e. K = K + NI, NI = I - N
	MatDiagonalScale(Kstg,Nstg,Nstg);
	VecDuplicate(Nstg,&NI);
	VecSet(NI,1.0);
	VecAXPY(NI,-1.0,Nstg);
	MatDiagonalSet(Kstg,NI,ADD_VALUES);
	
	// Setup the solver
	if (kspSTG==NULL){
		ierr = SetUpSolverSTG(opt);
		CHKERRQ(ierr);
	}
	ierr = KSPSetOperators(kspSTG,Kstg,Kstg); CHKERRQ(ierr);
	KSPSetUp(kspSTG);
	
	// Generate self-weight loadcase
	ierr = AssembleLoadSTG(opt,slc); CHKERRQ(ierr);
	
	// Zero out possible loads in the RHS that coincide with Dirichlet conditions
	VecPointwiseMult(RHSstg[slc],RHSstg[slc],Nstg);
	
	// Solve
	ierr = KSPSolve(kspSTG,RHSstg[slc],Ustg[slc]); CHKERRQ(ierr);
	
	// {
		// PetscViewer viewerVTS;
		// MPI_Comm comm;
		// comm = PETSC_COMM_WORLD;
		// ierr = PetscViewerVTKOpen(comm,"../output/ustg_check.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		// ierr = VecView(Ustg[slc],viewerVTS); CHKERRQ(ierr); // View vector
		// ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
	// }
	
	// Get iteration number and residual from KSP
	KSPGetIterationNumber(kspSTG,&niter);
	KSPGetResidualNorm(kspSTG,&rnorm);
	t2 = MPI_Wtime();
	PetscPrintf(PETSC_COMM_WORLD,"State solver, SLC %i:  iter: %i, rerr.: %e, time: %f\n",slc+1,niter,rnorm,t2-t1);
	t1 = t2;
	
	VecDestroy(&NI);
	
	return ierr;

}

PetscErrorCode LinearElasticity::ComputeObjectiveConstraintsSensitivities(TopOpt *opt) {

	// Error code
	PetscErrorCode ierr;

	// Solve state eqs
	ierr = SolveState(opt); CHKERRQ(ierr);

	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	ierr = DMDAGetElements_3D(opt->da_nodes,&nel,&nen,&necon); CHKERRQ(ierr);
	//DMDAGetElements(da_nodes,&nel,&nen,&necon); // Still issue with elemtype change !

	// Get pointer to the densities
	// Only x1 field here
	PetscScalar *xero;
	VecGetArray(opt->xEro1,&xero);

	// Get Solution
	std::vector<Vec> Uloc; // Array of local U vectors
	std::vector<PetscScalar*> up;
	
	Uloc.resize(opt->tnlc);
	up.resize(opt->tnlc);
	
	for(PetscInt i=0; i<opt->tnlc; i++){
		ierr = DMCreateLocalVector(opt->da_nodes,&Uloc[i]); CHKERRQ(ierr);
		VecSet(Uloc[i],0.0);
	}
	
	for(PetscInt i=0; i<opt->tnlc; i++){
		DMGlobalToLocalBegin(opt->da_nodes,U[i],INSERT_VALUES,Uloc[i]);
		DMGlobalToLocalEnd(opt->da_nodes,U[i],INSERT_VALUES,Uloc[i]);
		VecGetArray(Uloc[i],&up[i]);
	}
	
	// Get pointer to dfdx
	VecSet(opt->dfdx,0.0);
	PetscScalar *df;
	VecGetArray(opt->dfdx,&df);

	// edof array
	PetscInt edof[24];
	
	// For setting the local stiffness matrix
	PetscScalar a = opt->dx; // x-side length
	PetscScalar b = opt->dy; // y-side length
	PetscScalar c = opt->dz; // z-side length
	PetscScalar X[8] = {0.0, a, a, 0.0, 0.0, a, a, 0.0};
	PetscScalar Y[8] = {0.0, 0.0, b, b, 0.0, 0.0, b, b};
	PetscScalar Z[8] = {0.0, 0.0, 0.0, 0.0, c, c, c, c};
	PetscScalar ke[24*24];
	PetscScalar dke[24*24];
	
	// Zeroize objective and constraints
	opt->fx = 0.0;
	
	///////////////////////////////////////////////////
	// Loop over elements to get local contributions //
	///////////////////////////////////////////////////
	for (PetscInt i=0;i<nel;i++){

		// loop over element nodes
		for (PetscInt j=0;j<nen;j++){
			// Get local dofs
			for (PetscInt k=0;k<3;k++){
				edof[j*3+k] = 3*necon[i*nen+j]+k;
			}
		}
		
		PetscScalar uKu=0.0;
		if(opt->interprule == 1){ // Use SIMP for stiffness interpolation
			for (PetscInt k=0;k<24;k++){
				for (PetscInt h=0;h<24;h++){
					for (PetscInt l=0; l<opt->tnlc; l++){
						uKu += opt->weights[l]*up[l][edof[k]]*KE[k*24+h]*up[l][edof[h]];
					}		
				}
			}
			// Standard SIMP
			// Add uKU to objective, eroded 
			opt->fx += (opt->Emin + PetscPowScalar(xero[i],opt->penal)*(opt->Emax - opt->Emin))*uKu;
			// Set the sensitivity duKu, eroded
			df[2*i] = -1.0 * opt->penal*PetscPowScalar(xero[i],opt->penal-1)*(opt->Emax - opt->Emin)*uKu;
			
			// OTHER INTERPOLATIONS
			
			// // "LATTICE" SIMP
			// PetscScalar dens = xero[i];
			// PetscScalar E_of_rho = opt->Emin + (opt->Emax-opt->Emin)*(0.90735*PetscPowScalar(dens,3.0) 
				// - 0.04972*PetscPowScalar(dens,2.0) + 0.15411*dens);
			// PetscScalar dE_drho = (opt->Emax-opt->Emin)*(3.0*0.90735*PetscPowScalar(dens,2.0) 
				// - 2.0*0.04972*dens + 0.15411);
			// opt->fx += E_of_rho*uKu;
			// // Set the sensitivity duKu, eroded
			// df[i] = -1.0*dE_drho*uKu;
			
			// // HS bound
			// PetscScalar dens = xero[i];
			// PetscScalar E_of_rho = opt->Emin + (opt->Emax-opt->Emin)*(dens/(3.0-2.0*dens));
			// PetscScalar dE_drho = (opt->Emax-opt->Emin)*3.0/((3.0-2.0*dens)*(3.0-2.0*dens));
			// opt->fx += E_of_rho*uKu;
			// // Set the sensitivity duKu, eroded
			// df[i] = -1.0*dE_drho*uKu;
				
		
		
		}
		else{ 
			// Use lattice interpolation - not needed for the calculation of the final compliance
			// PetscScalar dens = xero[i];
			// Hex8Lattice(X, Y, Z, false, dens, opt->nu, opt->Emax, opt->Emin, ke);
			// Hex8LatticeDeriv(X, Y, Z, false, dens, opt->nu, opt->Emax, opt->Emin, dke);
			// PetscScalar duKu=0.0;
			// for (PetscInt k=0;k<24;k++){
				// for (PetscInt h=0;h<24;h++){
					// for (PetscInt l=0; l<opt->tnlc; l++){
						// uKu += opt->weights[l]*up[l][edof[k]]*ke[k*24+h]*up[l][edof[h]];
						// duKu += opt->weights[l]*up[l][edof[k]]*dke[k*24+h]*up[l][edof[h]];
					// }		
				// }
			// }
			// // Add uKU to objective, eroded 
			// opt->fx += uKu;
			// // Set the sensitivity duKu, eroded
			// df[2*i] = -duKu;
		}
	}

	// Allreduce fx[0]
	PetscScalar tmp=opt->fx;
	opt->fx=0.0;
	MPI_Allreduce(&tmp,&(opt->fx),1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
	
	// Compute volume constraints gx
	opt->gx[0] = 0.0;
	opt->gx[1] = 0.0;
	VecSet(opt->dgdx[0],0.0);
	VecSet(opt->dgdx[1],0.0);
	
	// g_0: dilated w.r.t. x1 only 
	VecSum(opt->xDil1, &(opt->gx[0]));
	opt->gx[0]=opt->gx[0]/(((PetscScalar)opt->nvoxel)*opt->volfrac1dil)-1.0;
	PetscScalar dvval1 = 1.0/(((PetscScalar)opt->nvoxel)*opt->volfrac1dil);
	ierr = VecStrideSet(opt->dgdx[0],0,dvval1); // Insert dvol/dx1 into dgdx[0]
	
	// g_1: filtered w.r.t. x2 only
	VecSum(opt->xTilde2, &(opt->gx[1]));
	opt->gx[1]=opt->gx[1]/(((PetscScalar)opt->nvoxel)*opt->volfrac2)-1.0;
	PetscScalar dvval2 = 1.0/(((PetscScalar)opt->nvoxel)*opt->volfrac2);
	ierr = VecStrideSet(opt->dgdx[1],1,dvval2); // Insert dvol/dx2 into dgdx[1]

	VecRestoreArray(opt->xEro1,&xero);
	VecRestoreArray(opt->dfdx,&df);
	
	for(PetscInt i=0; i<opt->tnlc; i++){
		VecRestoreArray(Uloc[i],&up[i]);
		VecDestroy(&Uloc[i]);
	}
	
	return(ierr);

}

PetscErrorCode LinearElasticity::ComputeObjectiveSensitivitiesSTG(TopOpt *opt) {
	
	// Error code
	PetscErrorCode ierr;
	
	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	ierr = DMDAGetElements_3D(opt->da_nodes,&nel,&nen,&necon); CHKERRQ(ierr);
	
	// Solve state equations for each stage separately
	for(PetscInt i=0; i<opt->nslc; i++){
		ierr = SolveStateSTG(opt,i); CHKERRQ(ierr);
	}
	
	// Get pointer to the densities and input of slices
	std::vector<PetscScalar*> xpstiff1, xpstiff2;
	xpstiff1.resize(opt->nslc);
	xpstiff2.resize(opt->nslc);
	
	std::vector<PetscScalar*> slcInputpload;
	slcInputpload.resize(opt->nslc);
	
	for(PetscInt i=0; i<opt->nslc; i++){
		VecGetArray(opt->xSTGstiff1[i],&xpstiff1[i]);
		VecGetArray(opt->xSTGstiff2[i],&xpstiff2[i]);
		VecGetArray(opt->slcInputload[i],&slcInputpload[i]);
	}
	
	// Get Solution
	std::vector<Vec> Uloc; // Array of local U vectors
	std::vector<PetscScalar*> up; 
	Uloc.resize(opt->nslc);
	up.resize(opt->nslc);
	
	for(PetscInt i=0; i<opt->nslc; i++){
		ierr = DMCreateLocalVector(opt->da_nodes,&Uloc[i]); CHKERRQ(ierr);
		VecSet(Uloc[i],0.0);
	}
	
	for(PetscInt i=0; i<opt->nslc; i++){
		DMGlobalToLocalBegin(opt->da_nodes,Ustg[i],INSERT_VALUES,Uloc[i]);
		DMGlobalToLocalEnd(opt->da_nodes,Ustg[i],INSERT_VALUES,Uloc[i]);
		VecGetArray(Uloc[i],&up[i]);
	}
	
	// Get pointer to dfdx
	VecSet(opt->dfdxSTG,0.0);
	PetscScalar *df;
	VecGetArray(opt->dfdxSTG,&df);
	
	// edof array
	PetscInt edof[24];
	
	// For setting the local stiffness matrix
	PetscScalar a = opt->dx; // x-side length
	PetscScalar b = opt->dy; // y-side length
	PetscScalar c = opt->dz; // z-side length
	PetscScalar X[8] = {0.0, a, a, 0.0, 0.0, a, a, 0.0};
	PetscScalar Y[8] = {0.0, 0.0, b, b, 0.0, 0.0, b, b};
	PetscScalar Z[8] = {0.0, 0.0, 0.0, 0.0, c, c, c, c};
	PetscScalar ke[24*24];
	PetscScalar dke[24*24];

	// Zeroize objective and constraints
	opt->compSTG = 0.0;
	
	//////////////////////////////////////////////////////////////
	// Loop over elements and slices to get local contributions //
	//////////////////////////////////////////////////////////////
	
	for (PetscInt i=0;i<nel;i++){

		// loop over element nodes
		for (PetscInt j=0;j<nen;j++){
			// Get local dofs
			for (PetscInt k=0;k<3;k++){
				edof[j*3+k] = 3*necon[i*nen+j]+k;
			}
		}
		
		// Collect uKu and duKu + dfu, all stages
		PetscScalar uKu = 0.0;
		PetscScalar duKudx1 = 0.0;
		PetscScalar duKudx2 = 0.0;
		PetscScalar dfu = 0.0;
		
		for (PetscInt k=0;k<24;k++){
			for (PetscInt h=0;h<24;h++){
				for (PetscInt l=0; l<opt->nslc; l++){
					
					// Collect compliance contribtuions, E(x1,x2)*uKu
					PetscScalar Estg1, Estg2, Estg;
					// Use SIMP for stiffness interpolation of x1
					Estg1 = opt->Emin + PetscPowScalar(xpstiff1[l][i],opt->penal)*(opt->Emax-opt->Emin);
					// Use lattice rule for stiffness interpolation of x2
					Estg2 = opt->Emin + (opt->Emax-opt->Emin)*(0.9*PetscPowScalar(xpstiff2[l][i],3.0) 
						- 0.0376*PetscPowScalar(xpstiff2[l][i],2.0) + 0.1513*xpstiff2[l][i]);
					// Avoid intersection
					Estg = Estg1 + (1.0-xpstiff1[l][i])*Estg2;
					// Collect derivatives w.r.t. x1, x2
					PetscScalar dEstgdx1, dEstgdx2;
					dEstgdx1 = opt->penal*PetscPowScalar(xpstiff1[l][i],opt->penal-1)*(opt->Emax-opt->Emin)
						- Estg2;
					dEstgdx2 = (opt->Emax-opt->Emin)*(3.0*0.9*PetscPowScalar(xpstiff2[l][i],2.0) 
						- 2.0*0.0376*xpstiff2[l][i] + 0.1513)*(1.0-xpstiff1[l][i]);
						
					uKu += Estg*up[l][edof[k]]*KE[k*24+h]*up[l][edof[h]];
					duKudx1 += -dEstgdx1*up[l][edof[k]]*KE[k*24+h]*up[l][edof[h]];
					duKudx2 += -dEstgdx2*up[l][edof[k]]*KE[k*24+h]*up[l][edof[h]];
				}
			}
			
			for (PetscInt l=0; l<opt->nslc; l++){
				if (k%3 == opt->printdir){
					if (slcInputpload[l][i] > 0.0){
						dfu += -2.0*0.125*up[l][edof[k]];
					}
				}
			}	
		}		
		
		// Add to compliance of staged construction
		opt->compSTG += uKu;
		// Set the Sensitivity 
        df[2*i] = duKudx1 + dfu; // x1 affects stiffness and load
		df[2*i+1] = duKudx2; // x2 affects stiffness only			
		
	}
	
	// Allreduce compSTG
	PetscScalar tmp = opt->compSTG;
	opt->compSTG=0.0;
	MPI_Allreduce(&tmp,&(opt->compSTG),1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);

	for(PetscInt i=0; i<opt->nslc; i++){
		VecRestoreArray(opt->xSTGstiff1[i],&xpstiff1[i]);
		VecRestoreArray(opt->xSTGstiff2[i],&xpstiff2[i]);
		VecRestoreArray(opt->slcInputload[i],&slcInputpload[i]);
	}
	
	VecRestoreArray(opt->dfdxSTG,&df);
	
	for(PetscInt i=0; i<opt->nslc; i++){
		VecRestoreArray(Uloc[i],&up[i]);
		VecDestroy(&Uloc[i]);
	}
	
	return(ierr);

}

//##################################################################
//##################################################################
//##################################################################
//######################### PRIVATE ################################
//##################################################################
//##################################################################

PetscErrorCode LinearElasticity::AssembleStiffnessMatrix(TopOpt *opt){

	PetscErrorCode ierr;

	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	ierr = DMDAGetElements_3D(opt->da_nodes,&nel,&nen,&necon);
	CHKERRQ(ierr);

	// Get pointer to the densities
	// For regular load cases, only x1 is used for stiffness
	PetscScalar *xp;
	VecGetArray(opt->xEro1,&xp);

	// Zero the matrix
	MatZeroEntries(K);

	// Edof array
	PetscInt edof[24];
	PetscScalar ke[24*24];
	
	// For setting the local stiffness matrix
	PetscScalar a = opt->dx; // x-side length
	PetscScalar b = opt->dy; // y-side length
	PetscScalar c = opt->dz; // z-side length
	PetscScalar X[8] = {0.0, a, a, 0.0, 0.0, a, a, 0.0};
	PetscScalar Y[8] = {0.0, 0.0, b, b, 0.0, 0.0, b, b};
	PetscScalar Z[8] = {0.0, 0.0, 0.0, 0.0, c, c, c, c};

	// Loop over elements
	for (PetscInt i=0;i<nel;i++){
		// loop over element nodes
		for (PetscInt j=0;j<nen;j++){
			// Get local dofs
			for (PetscInt k=0;k<3;k++){
				edof[j*3+k] = 3*necon[i*nen+j]+k;
			}
		}
		if(opt->interprule == 1){ 
		
			// Use SIMP for stiffness interpolation 
			PetscScalar dens = opt->Emin + PetscPowScalar(xp[i],opt->penal)*(opt->Emax-opt->Emin);
			
			// OTHER INTERPOLATIONS
			
			// PetscScalar dens = opt->Emin + (opt->Emax-opt->Emin)*(0.90735*PetscPowScalar(xp[i],3.0) 
				// - 0.04972*PetscPowScalar(xp[i],2.0) + 0.15411*xp[i]);
			
			// // HS bound
			// PetscScalar dens = opt->Emin + (opt->Emax-opt->Emin)*(xp[i]/(3.0-2.0*xp[i]));
			
			for (PetscInt k=0;k<24*24;k++){
				ke[k]=KE[k]*dens;
			}		
		}
		else{ 
			// Use lattice interpolation  <---- This option is not relevant for the final compliance calc. It has to be removed if orthotropic formulation will take place
			// PetscScalar dens = xp[i];
			// Hex8Lattice(X, Y, Z, false, dens, opt->nu, opt->Emax, opt->Emin, ke);
		}
		// Add values to the sparse matrix
		ierr = MatSetValuesLocal(K,24,edof,24,edof,ke,ADD_VALUES);
		CHKERRQ(ierr);
	}
	MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);
	
	VecRestoreArray(opt->xEro1,&xp);
	DMDARestoreElements(opt->da_nodes,&nel,&nen,&necon);

	return ierr;
}

PetscErrorCode LinearElasticity::AssembleStiffnessMatrixSTG(TopOpt *opt, PetscInt slc){

	PetscErrorCode ierr;

	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	ierr = DMDAGetElements_3D(opt->da_nodes,&nel,&nen,&necon);
	CHKERRQ(ierr);

	// Get pointer to the densities
	// For stages, both final structure and supports are used
	PetscScalar *xp1, *xp2;
	VecGetArray(opt->xSTGstiff1[slc],&xp1);
	VecGetArray(opt->xSTGstiff2[slc],&xp2);

	// Zero the matrix
	MatZeroEntries(Kstg);

	// Edof array
	PetscInt edof[24];
	PetscScalar ke[24*24];
	
	// For setting the local stiffness matrix
	PetscScalar a = opt->dx; // x-side length
	PetscScalar b = opt->dy; // y-side length
	PetscScalar c = opt->dz; // z-side length
	PetscScalar X[8] = {0.0, a, a, 0.0, 0.0, a, a, 0.0};
	PetscScalar Y[8] = {0.0, 0.0, b, b, 0.0, 0.0, b, b};
	PetscScalar Z[8] = {0.0, 0.0, 0.0, 0.0, c, c, c, c};

	// Loop over elements
	for (PetscInt i=0;i<nel;i++){
		// loop over element nodes
		for (PetscInt j=0;j<nen;j++){
			// Get local dofs
			for (PetscInt k=0;k<3;k++){
				edof[j*3+k] = 3*necon[i*nen+j]+k;
			}
		}
		if(opt->interprule == 1){ 
		
			// Use SIMP for stiffness interpolation of x1
			PetscScalar dens1 = opt->Emin + PetscPowScalar(xp1[i],opt->penal)*(opt->Emax-opt->Emin);
			
			// Use lattice rule for stiffness interpolation of x2
			PetscScalar x2 = (1.0-xp1[i])*xp2[i];
			PetscScalar dens2 = opt->Emin + (opt->Emax-opt->Emin)*(0.9*PetscPowScalar(x2,3.0) 
				- 0.0376*PetscPowScalar(x2,2.0) + 0.1513*x2);		
			// // HS bound
			// PetscScalar dens = opt->Emin + (opt->Emax-opt->Emin)*(xp[i]/(3.0-2.0*xp[i]));
			
			for (PetscInt k=0;k<24*24;k++){
				ke[k]=KE[k]*(dens1+dens2);
			}		
		}
		else{ 
			// Use lattice interpolation - when orthotropic rule is formulated use this section.
			// PetscScalar dens = xp[i];
			// Hex8Lattice(X, Y, Z, false, dens, opt->nu, opt->Emax, opt->Emin, ke);
		}
		// Add values to the sparse matrix
		ierr = MatSetValuesLocal(Kstg,24,edof,24,edof,ke,ADD_VALUES);
		CHKERRQ(ierr);
	}
	MatAssemblyBegin(Kstg,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Kstg,MAT_FINAL_ASSEMBLY);

	VecRestoreArray(opt->xSTGstiff1[slc],&xp1);
	VecRestoreArray(opt->xSTGstiff2[slc],&xp2);
	DMDARestoreElements(opt->da_nodes,&nel,&nen,&necon);

	return ierr;
}

PetscErrorCode LinearElasticity::AssembleLoadSTG(TopOpt *opt, PetscInt slc){

	PetscErrorCode ierr;

	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	ierr = DMDAGetElements_3D(opt->da_nodes,&nel,&nen,&necon); CHKERRQ(ierr);

	// Get pointer to the densities
	PetscScalar *xp;
	VecGetArray(opt->xSTGload[slc],&xp);

	// Zero the vector
	VecSet(RHSstg[slc],0.0);

	// Edof array
	PetscInt edof[24];
	PetscScalar rhse[24];

	// Loop over elements
	for (PetscInt i=0;i<nel;i++){
		
		// Loop over element nodes
		for (PetscInt j=0;j<nen;j++){
			// Get local dofs
			for (PetscInt k=0;k<3;k++){
				edof[j*3+k] = 3*necon[i*nen+j]+k;
			}
		}
		
		// Self-weight depends on the density in xSTG[slc]
		// Load applied only in the -dir direction
		for (PetscInt k=0;k<24;k++){
			rhse[k] = 0.0;
			if (k%3 == opt->printdir){
				rhse[k]=-0.125*xp[i];
			}
		}
		// Add values to the vector
		ierr = VecSetValuesLocal(RHSstg[slc],24,edof,rhse,ADD_VALUES); CHKERRQ(ierr);
	}
	
	VecAssemblyBegin(RHSstg[slc]);
	VecAssemblyEnd(RHSstg[slc]);

	VecRestoreArray(opt->xSTGload[slc],&xp);
	DMDARestoreElements(opt->da_nodes,&nel,&nen,&necon);

	return ierr;
	
}

PetscErrorCode LinearElasticity::SetUpSolver(TopOpt *opt){

	PetscErrorCode ierr;
	PC pc;

	// The fine grid Krylov method
	KSPCreate(PETSC_COMM_WORLD,&(ksp));

	// SET THE DEFAULT SOLVER PARAMETERS
	// The fine grid solver settings
	PetscScalar rtol = 1.0e-5;
	PetscScalar atol = 1.0e-50;
	PetscScalar dtol = 1.0e3;
	PetscInt restart = 100;
	PetscInt maxitsGlobal = 200;

	// Coarsegrid solver
	PetscScalar coarse_rtol = 1.0e-8;
	PetscScalar coarse_atol = 1.0e-50;
	PetscScalar coarse_dtol = 1e3;
	PetscInt coarse_maxits = 30;
	PetscInt coarse_restart = 30;

	// Number of smoothening iterations per up/down smooth_sweeps
	PetscInt smooth_sweeps = 4;

	// Set up the solver
	ierr = KSPSetType(ksp,KSPFGMRES); // KSPCG, KSPGMRESK
	CHKERRQ(ierr);

	ierr = KSPGMRESSetRestart(ksp,restart);
	CHKERRQ(ierr);

	ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxitsGlobal);
	CHKERRQ(ierr);

	ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE); 
	CHKERRQ(ierr);

	ierr = KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
	CHKERRQ(ierr);

	ierr = KSPSetOperators(ksp,K,K);
	CHKERRQ(ierr);

	// The preconditinoer
	KSPGetPC(ksp,&pc);
	// Make PCMG the default solver
	PCSetType(pc,PCMG);

	// Set solver from options
	KSPSetFromOptions(ksp);

	// Get the prec again - check if it has changed
	KSPGetPC(ksp,&pc);

	// Flag for pcmg pc
	PetscBool pcmg_flag = PETSC_TRUE;
	PetscObjectTypeCompare((PetscObject)pc,PCMG,&pcmg_flag);

	// Only if PCMG is used
	if (pcmg_flag){

		// DMs for grid hierachy
		DM  *da_list,*daclist;
		Mat R;

		PetscMalloc(sizeof(DM)*opt->nlvls,&da_list);
		for (PetscInt k=0; k<opt->nlvls; k++) da_list[k] = NULL;
		PetscMalloc(sizeof(DM)*opt->nlvls,&daclist);
		for (PetscInt k=0; k<opt->nlvls; k++) daclist[k] = NULL;

		// Set 0 to the finest level
		daclist[0] = opt->da_nodes;

		// Coordinates
		PetscReal xmin=opt->xc[0], xmax=opt->xc[1], ymin=opt->xc[2], ymax=opt->xc[3], zmin=opt->xc[4], zmax=opt->xc[5];

		// Set up the coarse meshes
		DMCoarsenHierarchy(opt->da_nodes,opt->nlvls-1,&daclist[1]);
		for (PetscInt k=0; k<opt->nlvls; k++) {
			// NOTE: finest grid is nlevels - 1: PCMG MUST USE THIS ORDER ???
			da_list[k] = daclist[opt->nlvls-1-k];
			// THIS SHOULD NOT BE NECESSARY
			DMDASetUniformCoordinates(da_list[k],xmin,xmax,ymin,ymax,zmin,zmax);
		}

		// the PCMG specific options
		PCMGSetLevels(pc,opt->nlvls,NULL);
		PCMGSetType(pc,PC_MG_MULTIPLICATIVE); // Default
		PCMGSetCycleType(pc,PC_MG_CYCLE_V);
		PCMGSetGalerkin(pc,PETSC_TRUE);
		for (PetscInt k=1; k<opt->nlvls; k++) {
			DMCreateInterpolation(da_list[k-1],da_list[k],&R,NULL);
			PCMGSetInterpolation(pc,k,R);
			MatDestroy(&R);
		}

		// tidy up
		for (PetscInt k=1; k<opt->nlvls; k++) { // DO NOT DESTROY LEVEL 0
			DMDestroy(&daclist[k]);
		}
		PetscFree(da_list);
		PetscFree(daclist);

		// AVOID THE DEFAULT FOR THE MG PART
		{
			// SET the coarse grid solver:
			// i.e. get a pointer to the ksp and change its settings
			KSP cksp;
			PCMGGetCoarseSolve(pc,&cksp);
			
			// The solver
			ierr = KSPSetType(cksp,KSPGMRES); // KSPCG, KSPFGMRES

			ierr = KSPGMRESSetRestart(cksp,coarse_restart);

			ierr = KSPSetTolerances(cksp,coarse_rtol,coarse_atol,coarse_dtol,coarse_maxits);
			// The preconditioner
			PC cpc;
			KSPGetPC(cksp,&cpc);
			PCSetType(cpc,PCSOR); // PCSOR, PCSPAI (NEEDS TO BE COMPILED), PCJACOBI
			
			// Set smoothers on all levels (except for coarse grid):
			for (PetscInt k=1;k<opt->nlvls;k++){
				KSP dksp;
				PCMGGetSmoother(pc,k,&dksp);
				PC dpc;
				KSPGetPC(dksp,&dpc);
				ierr = KSPSetType(dksp,KSPGMRES); // KSPCG, KSPGMRES, KSPCHEBYSHEV (VERY GOOD FOR SPD)

				ierr = KSPGMRESSetRestart(dksp,smooth_sweeps);
				ierr = KSPSetTolerances(dksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,smooth_sweeps); // NOTE in the above maxitr=restart;
				PCSetType(dpc,PCSOR);// PCJACOBI, PCSOR for KSPCHEBYSHEV very good
			}
		}
	}

	// Write check to screen:
	// Check the overall Krylov solver
	KSPType ksptype;
	KSPGetType(ksp,&ksptype);
	PCType pctype;
	PCGetType(pc,&pctype);
	PetscInt mmax;
	KSPGetTolerances(ksp,NULL,NULL,NULL,&mmax);
	PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");
	PetscPrintf(PETSC_COMM_WORLD,"################# Linear solver settings #####################\n");
	PetscPrintf(PETSC_COMM_WORLD,"# Main solver: %s, prec.: %s, maxiter.: %i \n",ksptype,pctype,mmax);

	// Only if pcmg is used
	if (pcmg_flag){
		// Check the smoothers and coarse grid solver:
		for (PetscInt k=0;k<opt->nlvls;k++){
			KSP dksp;
			PC dpc;
			KSPType dksptype;
			PCMGGetSmoother(pc,k,&dksp);
			KSPGetType(dksp,&dksptype);
			KSPGetPC(dksp,&dpc);
			PCType dpctype;
			PCGetType(dpc,&dpctype);
			PetscInt mmax;
			KSPGetTolerances(dksp,NULL,NULL,NULL,&mmax);
			PetscPrintf(PETSC_COMM_WORLD,"# Level %i smoother: %s, prec.: %s, sweep: %i \n",k,dksptype,dpctype,mmax);
		}
	}
	PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");


	return(ierr);
}

PetscErrorCode LinearElasticity::SetUpSolverSTG(TopOpt *opt){

	PetscErrorCode ierr;
	PC pc;

	// The fine grid Krylov method
	KSPCreate(PETSC_COMM_WORLD,&(kspSTG));

	// SET THE DEFAULT SOLVER PARAMETERS
	// The fine grid solver settings
	PetscScalar rtol = 1.0e-5;
	PetscScalar atol = 1.0e-50;
	PetscScalar dtol = 1.0e3;
	PetscInt restart = 100;
	PetscInt maxitsGlobal = 200;

	// Coarsegrid solver
	PetscScalar coarse_rtol = 1.0e-8;
	PetscScalar coarse_atol = 1.0e-50;
	PetscScalar coarse_dtol = 1e3;
	PetscInt coarse_maxits = 30;
	PetscInt coarse_restart = 30;

	// Number of smoothening iterations per up/down smooth_sweeps
	PetscInt smooth_sweeps = 4;

	// Set up the solver
	ierr = KSPSetType(kspSTG,KSPFGMRES); // KSPCG, KSPGMRESK
	CHKERRQ(ierr);

	ierr = KSPGMRESSetRestart(kspSTG,restart);
	CHKERRQ(ierr);

	ierr = KSPSetTolerances(kspSTG,rtol,atol,dtol,maxitsGlobal);
	CHKERRQ(ierr);

	ierr = KSPSetInitialGuessNonzero(kspSTG,PETSC_TRUE); /////!!!!!!!!
	CHKERRQ(ierr);

	ierr = KSPSetNormType(kspSTG,KSP_NORM_UNPRECONDITIONED);
	CHKERRQ(ierr);

	ierr = KSPSetOperators(kspSTG,Kstg,Kstg);
	CHKERRQ(ierr);

	// The preconditinoer
	KSPGetPC(kspSTG,&pc);
	// Make PCMG the default solver
	PCSetType(pc,PCMG);

	// Set solver from options
	KSPSetFromOptions(kspSTG);

	// Get the prec again - check if it has changed
	KSPGetPC(kspSTG,&pc);

	// Flag for pcmg pc
	PetscBool pcmg_flag = PETSC_TRUE;
	PetscObjectTypeCompare((PetscObject)pc,PCMG,&pcmg_flag);

	// Only if PCMG is used
	if (pcmg_flag){

		// DMs for grid hierachy
		DM  *da_list,*daclist;
		Mat R;

		PetscMalloc(sizeof(DM)*opt->nlvls,&da_list);
		for (PetscInt k=0; k<opt->nlvls; k++) da_list[k] = NULL;
		PetscMalloc(sizeof(DM)*opt->nlvls,&daclist);
		for (PetscInt k=0; k<opt->nlvls; k++) daclist[k] = NULL;

		// Set 0 to the finest level
		daclist[0] = opt->da_nodes;

		// Coordinates
		PetscReal xmin=opt->xc[0], xmax=opt->xc[1], ymin=opt->xc[2], ymax=opt->xc[3], zmin=opt->xc[4], zmax=opt->xc[5];

		// Set up the coarse meshes
		DMCoarsenHierarchy(opt->da_nodes,opt->nlvls-1,&daclist[1]);
		for (PetscInt k=0; k<opt->nlvls; k++) {
			// NOTE: finest grid is nlevels - 1: PCMG MUST USE THIS ORDER ???
			da_list[k] = daclist[opt->nlvls-1-k];
			// THIS SHOULD NOT BE NECESSARY
			DMDASetUniformCoordinates(da_list[k],xmin,xmax,ymin,ymax,zmin,zmax);
		}

		// the PCMG specific options
		PCMGSetLevels(pc,opt->nlvls,NULL);
		PCMGSetType(pc,PC_MG_MULTIPLICATIVE); // Default
		PCMGSetCycleType(pc,PC_MG_CYCLE_V);
		PCMGSetGalerkin(pc,PETSC_TRUE);
		for (PetscInt k=1; k<opt->nlvls; k++) {
			DMCreateInterpolation(da_list[k-1],da_list[k],&R,NULL);
			PCMGSetInterpolation(pc,k,R);
			MatDestroy(&R);
		}

		// tidy up
		for (PetscInt k=1; k<opt->nlvls; k++) { // DO NOT DESTROY LEVEL 0
			DMDestroy(&daclist[k]);
		}
		PetscFree(da_list);
		PetscFree(daclist);

		// AVOID THE DEFAULT FOR THE MG PART
		{
			// SET the coarse grid solver:
			// i.e. get a pointer to the ksp and change its settings
			KSP cksp;
			PCMGGetCoarseSolve(pc,&cksp);
			
			// The solver
			ierr = KSPSetType(cksp,KSPGMRES); // KSPCG, KSPFGMRES

			ierr = KSPGMRESSetRestart(cksp,coarse_restart);

			ierr = KSPSetTolerances(cksp,coarse_rtol,coarse_atol,coarse_dtol,coarse_maxits);
			// The preconditioner
			PC cpc;
			KSPGetPC(cksp,&cpc);
			PCSetType(cpc,PCSOR); // PCSOR, PCSPAI (NEEDS TO BE COMPILED), PCJACOBI
			
			// Set smoothers on all levels (except for coarse grid):
			for (PetscInt k=1;k<opt->nlvls;k++){
				KSP dksp;
				PCMGGetSmoother(pc,k,&dksp);
				PC dpc;
				KSPGetPC(dksp,&dpc);
				ierr = KSPSetType(dksp,KSPGMRES); // KSPCG, KSPGMRES, KSPCHEBYSHEV (VERY GOOD FOR SPD)
				ierr = KSPGMRESSetRestart(dksp,smooth_sweeps);
				ierr = KSPSetTolerances(dksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,smooth_sweeps); // NOTE in the above maxitr=restart;
				PCSetType(dpc,PCSOR);// PCJACOBI, PCSOR for KSPCHEBYSHEV very good
			}
		}
	}

	// Write check to screen:
	// Check the overall Krylov solver
	KSPType ksptype;
	KSPGetType(kspSTG,&ksptype);
	PCType pctype;
	PCGetType(pc,&pctype);
	PetscInt mmax;
	KSPGetTolerances(ksp,NULL,NULL,NULL,&mmax);
	PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");
	PetscPrintf(PETSC_COMM_WORLD,"################# Linear solver settings #####################\n");
	PetscPrintf(PETSC_COMM_WORLD,"# Staged solver: %s, prec.: %s, maxiter.: %i \n",ksptype,pctype,mmax);

	// Only if pcmg is used
	if (pcmg_flag){
		// Check the smoothers and coarse grid solver:
		for (PetscInt k=0;k<opt->nlvls;k++){
			KSP dksp;
			PC dpc;
			KSPType dksptype;
			PCMGGetSmoother(pc,k,&dksp);
			KSPGetType(dksp,&dksptype);
			KSPGetPC(dksp,&dpc);
			PCType dpctype;
			PCGetType(dpc,&dpctype);
			PetscInt mmax;
			KSPGetTolerances(dksp,NULL,NULL,NULL,&mmax);
			PetscPrintf(PETSC_COMM_WORLD,"# Level %i smoother: %s, prec.: %s, sweep: %i \n",k,dksptype,dpctype,mmax);
		}
	}
	PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");


	return(ierr);
}

PetscErrorCode LinearElasticity::DMDAGetElements_3D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[]) {
	PetscErrorCode ierr;
	DM_DA          *da = (DM_DA*)dm->data;
	PetscInt       i,xs,xe,Xs,Xe;
	PetscInt       j,ys,ye,Ys,Ye;
	PetscInt       k,zs,ze,Zs,Ze;
	PetscInt       cnt=0, cell[8], ns=1, nn=8;
	PetscInt       c;
	if (!da->e) {
		if (da->elementtype == DMDA_ELEMENT_Q1) {ns=1; nn=8;}
		ierr = DMDAGetCorners(dm,&xs,&ys,&zs,&xe,&ye,&ze);
		CHKERRQ(ierr);
		ierr = DMDAGetGhostCorners(dm,&Xs,&Ys,&Zs,&Xe,&Ye,&Ze);
		CHKERRQ(ierr);
		xe    += xs; Xe += Xs; if (xs != Xs) xs -= 1;
		ye    += ys; Ye += Ys; if (ys != Ys) ys -= 1;
		ze    += zs; Ze += Zs; if (zs != Zs) zs -= 1;
		da->ne = ns*(xe - xs - 1)*(ye - ys - 1)*(ze - zs - 1);
		PetscMalloc((1 + nn*da->ne)*sizeof(PetscInt),&da->e);
		for (k=zs; k<ze-1; k++) {
			for (j=ys; j<ye-1; j++) {
				for (i=xs; i<xe-1; i++) {
					cell[0] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
					cell[1] = (i-Xs+1) + (j-Ys  )*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
					cell[2] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
					cell[3] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
					cell[4] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
					cell[5] = (i-Xs+1) + (j-Ys  )*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
					cell[6] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
					cell[7] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
					if (da->elementtype == DMDA_ELEMENT_Q1) {
						for (c=0; c<ns*nn; c++) da->e[cnt++] = cell[c];
					}
				}
			}
		}
	}
	*nel = da->ne;
	*nen = nn;
	*e   = da->e;
	return(0);
}

PetscInt LinearElasticity::Hex8Isoparametric(PetscScalar *X, PetscScalar *Y, PetscScalar *Z, PetscScalar nu, PetscInt redInt, PetscScalar *ke, PetscScalar BMatrix[6][24]){
	// HEX8_ISOPARAMETRIC - Computes HEX8 isoparametric element matrices
	// The element stiffness matrix is computed as:
	//
	//       ke = int(int(int(B^T*C*B,x),y),z)
	//
	// For an isoparameteric element this integral becomes:
	//
	//       ke = int(int(int(B^T*C*B*det(J),xi=-1..1),eta=-1..1),zeta=-1..1)
	//
	// where B is the more complicated expression:
	// B = [dx*alpha1 + dy*alpha2 + dz*alpha3]*N
	// where
	// dx = [invJ11 invJ12 invJ13]*[dxi deta dzeta]
	// dy = [invJ21 invJ22 invJ23]*[dxi deta dzeta]
	// dy = [invJ31 invJ32 invJ33]*[dxi deta dzeta]
	//
	// Remark: The elasticity modulus is left out in the below
	// computations, because we multiply with it afterwards (the aim is
	// topology optimization).
	// Furthermore, this is not the most efficient code, but it is readable.
	//
	/////////////////////////////////////////////////////////////////////////////////
	//////// INPUT:
	// X, Y, Z  = Vectors containing the coordinates of the eight nodes
	//               (x1,y1,z1,x2,y2,z2,...,x8,y8,z8). Where node 1 is in the lower
	//               left corner, and node 2 is the next node counterclockwise
	//               (looking in the negative z-dir).
	//               Finish the x-y-plane and then move in the positive z-dir.
	// redInt   = Reduced integration option boolean (here an integer).
	//           	redInt == 0 (false): Full integration
	//           	redInt == 1 (true): Reduced integration
	// nu 		= Poisson's ratio.
	//
	//////// OUTPUT:
	// ke  = Element stiffness matrix. Needs to be multiplied with elasticity modulus
	//
	//   Written 2013 at
	//   Department of Mechanical Engineering
	//   Technical University of Denmark (DTU).
	/////////////////////////////////////////////////////////////////////////////////

	//// COMPUTE ELEMENT STIFFNESS MATRIX
	// Lame's parameters (with E=1.0):
	PetscScalar lambda = nu/((1.0+nu)*(1.0-2.0*nu));
	PetscScalar mu = 1.0/(2.0*(1.0+nu));
	// Constitutive matrix
	PetscScalar C[6][6] = {{lambda+2.0*mu, lambda, lambda, 0.0, 0.0, 0.0},
		{lambda, lambda+2.0*mu, lambda, 0.0, 0.0, 0.0},
		{lambda, lambda, lambda+2.0*mu, 0.0, 0.0, 0.0},
		{0.0,    0.0,    0.0,           mu,  0.0, 0.0},
		{0.0, 	0.0, 	0.0, 		   0.0, mu,  0.0},
		{0.0, 	0.0,	0.0, 		   0.0, 0.0, mu}};
	// Gauss points (GP) and weigths
	// Two Gauss points in all directions (total of eight)
	PetscScalar GP[2] = {-0.577350269189626, 0.577350269189626};
	// Corresponding weights
	PetscScalar W[2] = {1.0, 1.0};
	// If reduced integration only use one GP
	if (redInt){
		GP[1] = 0.0;
		W[1] = 2.0;
	}
	// Matrices that help when we gather the strain-displacement matrix:
	PetscScalar alpha1[6][3]; PetscScalar alpha2[6][3]; PetscScalar alpha3[6][3];
	memset(alpha1, 0, sizeof(alpha1[0][0])*6*3); // zero out
	memset(alpha2, 0, sizeof(alpha2[0][0])*6*3); // zero out
	memset(alpha3, 0, sizeof(alpha3[0][0])*6*3); // zero out
	alpha1[0][0] = 1.0; alpha1[3][1] = 1.0; alpha1[5][2] = 1.0;
	alpha2[1][1] = 1.0; alpha2[3][0] = 1.0; alpha2[4][2] = 1.0;
	alpha3[2][2] = 1.0; alpha3[4][1] = 1.0; alpha3[5][0] = 1.0;
	PetscScalar dNdxi[8]; PetscScalar dNdeta[8]; PetscScalar dNdzeta[8];
	PetscScalar J[3][3];
	PetscScalar invJ[3][3];
	PetscScalar beta[6][3];
	// PetscScalar B[6][24]; // Note: Small enough to be allocated on stack
	PetscScalar *dN;
	// Make sure the stiffness matrix is zeroed out:
	memset(ke, 0, sizeof(ke[0])*24*24);
	// Perform the numerical integration
	for (PetscInt ii=0; ii<2-redInt; ii++){
		for (PetscInt jj=0; jj<2-redInt; jj++){
			for (PetscInt kk=0; kk<2-redInt; kk++){
				// Integration point
				PetscScalar xi = GP[ii];
				PetscScalar eta = GP[jj];
				PetscScalar zeta = GP[kk];
				// Differentiated shape functions
				DifferentiatedShapeFunctions(xi, eta, zeta, dNdxi, dNdeta, dNdzeta);
				// Jacobian
				J[0][0] = Dot(dNdxi,X,8); J[0][1] = Dot(dNdxi,Y,8); J[0][2] = Dot(dNdxi,Z,8);
				J[1][0] = Dot(dNdeta,X,8); J[1][1] = Dot(dNdeta,Y,8); J[1][2] = Dot(dNdeta,Z,8);
				J[2][0] = Dot(dNdzeta,X,8); J[2][1] = Dot(dNdzeta,Y,8); J[2][2] = Dot(dNdzeta,Z,8);
				// Inverse and determinant
				PetscScalar detJ = Inverse3M(J, invJ);
				// Weight factor at this point
				PetscScalar weight = W[ii]*W[jj]*W[kk]*detJ;
				// Strain-displacement matrix
				memset(BMatrix, 0, sizeof(BMatrix[0][0])*6*24); // zero out
				for (PetscInt ll=0; ll<3; ll++){
					// Add contributions from the different derivatives
					if (ll==0) {dN = dNdxi;}
					if (ll==1) {dN = dNdeta;}
					if (ll==2) {dN = dNdzeta;}
					// Assemble strain operator
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<3; j++){
							beta[i][j] = invJ[0][ll]*alpha1[i][j]
								+invJ[1][ll]*alpha2[i][j]
								+invJ[2][ll]*alpha3[i][j];
						}
					}
					// Add contributions to strain-displacement matrix
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<24; j++){
							BMatrix[i][j] = BMatrix[i][j] + beta[i][j%3]*dN[j/3];
						}
					}
				}
				// Finally, add to the element matrix
				for (PetscInt i=0; i<24; i++){
					for (PetscInt j=0; j<24; j++){
						for (PetscInt k=0; k<6; k++){
							for (PetscInt l=0; l<6; l++){

								ke[j+24*i] = ke[j+24*i] + weight*(BMatrix[k][i] * C[k][l] * BMatrix[l][j]);
							}
						}
					}
				}
			}
		}
	}
	// Output BMatrix with reduced integration (stresses at mid point)
	for (PetscInt ii=0; ii<2-1; ii++){
		for (PetscInt jj=0; jj<2-1; jj++){
			for (PetscInt kk=0; kk<2-1; kk++){
				// Integration point - mid point of the element
				PetscScalar xi = 0.0;
				PetscScalar eta = 0.0;
				PetscScalar zeta = 0.0;
				// Differentiated shape functions
				DifferentiatedShapeFunctions(xi, eta, zeta, dNdxi, dNdeta, dNdzeta);
				// Jacobian
				J[0][0] = Dot(dNdxi,X,8); J[0][1] = Dot(dNdxi,Y,8); J[0][2] = Dot(dNdxi,Z,8);
				J[1][0] = Dot(dNdeta,X,8); J[1][1] = Dot(dNdeta,Y,8); J[1][2] = Dot(dNdeta,Z,8);
				J[2][0] = Dot(dNdzeta,X,8); J[2][1] = Dot(dNdzeta,Y,8); J[2][2] = Dot(dNdzeta,Z,8);
				// Inverse and determinant
				PetscScalar detJ = Inverse3M(J, invJ);
				// Strain-displacement matrix
				memset(BMatrix, 0, sizeof(BMatrix[0][0])*6*24); // zero out
				for (PetscInt ll=0; ll<3; ll++){
					// Add contributions from the different derivatives
					if (ll==0) {dN = dNdxi;}
					if (ll==1) {dN = dNdeta;}
					if (ll==2) {dN = dNdzeta;}
					// Assemble strain operator
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<3; j++){
							beta[i][j] = invJ[0][ll]*alpha1[i][j]
								+invJ[1][ll]*alpha2[i][j]
								+invJ[2][ll]*alpha3[i][j];
						}
					}
					// Add contributions to strain-displacement matrix
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<24; j++){
							BMatrix[i][j] = BMatrix[i][j] + beta[i][j%3]*dN[j/3];
						}
					}
				}
			}
		}
	}

	return 0;
}

PetscInt LinearElasticity::Hex8Lattice(PetscScalar *X, PetscScalar *Y, PetscScalar *Z, PetscInt redInt, PetscScalar dens, PetscScalar nu0, PetscScalar E0, PetscScalar Emin, PetscScalar *ke){
	
	//// COMPUTE ELEMENT STIFFNESS MATRIX
	
	// For lattices, C[][] will contain polynomials that depend on density
	// The values for solid
	PetscScalar G0 = E0/(2.0*(1.0+nu0));
	PetscScalar Gmin = Emin/(2.0*(1.0+nu0));
	// The values as function of density
	PetscScalar nu_of_rho = nu0*(0.83537*PetscPowScalar(dens,3.0) 
		- 0.83750*PetscPowScalar(dens,2.0) - 0.03676*dens 
		+ 1.0433);
	PetscScalar E_of_rho = Emin + E0*(0.90735*PetscPowScalar(dens,3.0) 
		- 0.04972*PetscPowScalar(dens,2.0) + 0.15411*dens 
		+ 0.0);
	PetscScalar G_of_rho = Gmin + G0*(0.34006*PetscPowScalar(dens,3.0) 
		+ 0.46330*PetscPowScalar(dens,2.0) + 0.20300*dens 
		+ 0.0);	
		
	// Lame's parameters 
	PetscScalar lambda = E_of_rho*nu_of_rho/((1.0+nu_of_rho)*(1.0-2.0*nu_of_rho));
	PetscScalar mu = G_of_rho; 
	// Constitutive matrix
	PetscScalar C[6][6] = {{lambda+2.0*mu, lambda, lambda, 0.0, 0.0, 0.0},
		{lambda, lambda+2.0*mu, lambda, 0.0, 0.0, 0.0},
		{lambda, lambda, lambda+2.0*mu, 0.0, 0.0, 0.0},
		{0.0,    0.0,    0.0,           mu,  0.0, 0.0},
		{0.0, 	0.0, 	0.0, 		   0.0, mu,  0.0},
		{0.0, 	0.0,	0.0, 		   0.0, 0.0, mu}};
	// Gauss points (GP) and weigths
	// Two Gauss points in all directions (total of eight)
	PetscScalar GP[2] = {-0.577350269189626, 0.577350269189626};
	// Corresponding weights
	PetscScalar W[2] = {1.0, 1.0};
	// If reduced integration only use one GP
	if (redInt){
		GP[1] = 0.0;
		W[1] = 2.0;
	}
	// Matrices that help when we gather the strain-displacement matrix:
	PetscScalar alpha1[6][3]; PetscScalar alpha2[6][3]; PetscScalar alpha3[6][3];
	memset(alpha1, 0, sizeof(alpha1[0][0])*6*3); // zero out
	memset(alpha2, 0, sizeof(alpha2[0][0])*6*3); // zero out
	memset(alpha3, 0, sizeof(alpha3[0][0])*6*3); // zero out
	alpha1[0][0] = 1.0; alpha1[3][1] = 1.0; alpha1[5][2] = 1.0;
	alpha2[1][1] = 1.0; alpha2[3][0] = 1.0; alpha2[4][2] = 1.0;
	alpha3[2][2] = 1.0; alpha3[4][1] = 1.0; alpha3[5][0] = 1.0;
	PetscScalar dNdxi[8]; PetscScalar dNdeta[8]; PetscScalar dNdzeta[8];
	PetscScalar J[3][3];
	PetscScalar invJ[3][3];
	PetscScalar beta[6][3];
	PetscScalar BMatrix[6][24]; // Note: Small enough to be allocated on stack
	PetscScalar *dN;
	// Make sure the stiffness matrix is zeroed out:
	memset(ke, 0, sizeof(ke[0])*24*24);
	// Perform the numerical integration
	for (PetscInt ii=0; ii<2-redInt; ii++){
		for (PetscInt jj=0; jj<2-redInt; jj++){
			for (PetscInt kk=0; kk<2-redInt; kk++){
				// Integration point
				PetscScalar xi = GP[ii];
				PetscScalar eta = GP[jj];
				PetscScalar zeta = GP[kk];
				// Differentiated shape functions
				DifferentiatedShapeFunctions(xi, eta, zeta, dNdxi, dNdeta, dNdzeta);
				// Jacobian
				J[0][0] = Dot(dNdxi,X,8); J[0][1] = Dot(dNdxi,Y,8); J[0][2] = Dot(dNdxi,Z,8);
				J[1][0] = Dot(dNdeta,X,8); J[1][1] = Dot(dNdeta,Y,8); J[1][2] = Dot(dNdeta,Z,8);
				J[2][0] = Dot(dNdzeta,X,8); J[2][1] = Dot(dNdzeta,Y,8); J[2][2] = Dot(dNdzeta,Z,8);
				// Inverse and determinant
				PetscScalar detJ = Inverse3M(J, invJ);
				// Weight factor at this point
				PetscScalar weight = W[ii]*W[jj]*W[kk]*detJ;
				// Strain-displacement matrix
				memset(BMatrix, 0, sizeof(BMatrix[0][0])*6*24); // zero out
				for (PetscInt ll=0; ll<3; ll++){
					// Add contributions from the different derivatives
					if (ll==0) {dN = dNdxi;}
					if (ll==1) {dN = dNdeta;}
					if (ll==2) {dN = dNdzeta;}
					// Assemble strain operator
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<3; j++){
							beta[i][j] = invJ[0][ll]*alpha1[i][j]
								+invJ[1][ll]*alpha2[i][j]
								+invJ[2][ll]*alpha3[i][j];
						}
					}
					// Add contributions to strain-displacement matrix
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<24; j++){
							BMatrix[i][j] = BMatrix[i][j] + beta[i][j%3]*dN[j/3];
						}
					}
				}
				// Finally, add to the element matrix
				for (PetscInt i=0; i<24; i++){
					for (PetscInt j=0; j<24; j++){
						for (PetscInt k=0; k<6; k++){
							for (PetscInt l=0; l<6; l++){

								ke[j+24*i] = ke[j+24*i] + weight*(BMatrix[k][i] * C[k][l] * BMatrix[l][j]);
							}
						}
					}
				}
			}
		}
	}

	return 0;
}

PetscInt LinearElasticity::Hex8LatticeDeriv(PetscScalar *X, PetscScalar *Y, PetscScalar *Z, PetscInt redInt, PetscScalar dens, PetscScalar nu0, PetscScalar E0, PetscScalar Emin, PetscScalar *dke){
	
	//// COMPUTE DERIVATIVE OF ELEMENT STIFFNESS MATRIX W.R.T. DENSITY
	
	// For lattices, C[][] will contain polynomials that depend on density
	// The values for solid
	PetscScalar G0 = E0/(2.0*(1.0+nu0));
	PetscScalar Gmin = Emin/(2.0*(1.0+nu0));
	// The values as function of density
	PetscScalar nu_of_rho = nu0*(0.83537*PetscPowScalar(dens,3.0) 
		- 0.83750*PetscPowScalar(dens,2.0) - 0.03676*dens 
		+ 1.0433);
	PetscScalar E_of_rho = Emin + E0*(0.90735*PetscPowScalar(dens,3.0) 
		- 0.04972*PetscPowScalar(dens,2.0) + 0.15411*dens 
		+ 0.0);
	// The derivatives of nu, E, G w.r.t. density
	PetscScalar dnu_drho = nu0*(3.0*0.83537*PetscPowScalar(dens,2.0) 
		- 2.0*0.83750*dens - 0.03676); 
	PetscScalar dE_drho = E0*(3.0*0.90735*PetscPowScalar(dens,2.0) 
		- 2.0*0.04972*dens + 0.15411);
	PetscScalar dG_drho = G0*(3.0*0.34006*PetscPowScalar(dens,2.0) 
		+ 2.0*0.46330*dens + 0.20300);
		
	// The derivatives of Lame's parameters 
	PetscScalar dlambda_dnu = E_of_rho*(2.0*PetscPowScalar(nu_of_rho,2.0)+1.0) / 
		((2.0*PetscPowScalar(nu_of_rho,2.0)+nu_of_rho-1.0)*(2.0*PetscPowScalar(nu_of_rho,2.0)+nu_of_rho-1.0));
	PetscScalar dlambda_dE = nu_of_rho/((1.0+nu_of_rho)*(1.0-2.0*nu_of_rho));
	PetscScalar dlambda_drho = dlambda_dE*dE_drho + dlambda_dnu*dnu_drho;
	PetscScalar dmu_drho = dG_drho; 
	
	// Derivative of constitutive matrix
	PetscScalar dC_drho[6][6] = {{dlambda_drho+2.0*dmu_drho, dlambda_drho, dlambda_drho, 0.0, 0.0, 0.0},
		{dlambda_drho, dlambda_drho+2.0*dmu_drho, dlambda_drho, 0.0, 0.0, 0.0},
		{dlambda_drho, dlambda_drho, dlambda_drho+2.0*dmu_drho, 0.0, 0.0, 0.0},
		{0.0,    0.0,    0.0,           dmu_drho,  0.0, 0.0},
		{0.0, 	0.0, 	0.0, 		   0.0, dmu_drho,  0.0},
		{0.0, 	0.0,	0.0, 		   0.0, 0.0, dmu_drho}};
	// Gauss points (GP) and weigths
	// Two Gauss points in all directions (total of eight)
	PetscScalar GP[2] = {-0.577350269189626, 0.577350269189626};
	// Corresponding weights
	PetscScalar W[2] = {1.0, 1.0};
	// If reduced integration only use one GP
	if (redInt){
		GP[1] = 0.0;
		W[1] = 2.0;
	}
	// Matrices that help when we gather the strain-displacement matrix:
	PetscScalar alpha1[6][3]; PetscScalar alpha2[6][3]; PetscScalar alpha3[6][3];
	memset(alpha1, 0, sizeof(alpha1[0][0])*6*3); // zero out
	memset(alpha2, 0, sizeof(alpha2[0][0])*6*3); // zero out
	memset(alpha3, 0, sizeof(alpha3[0][0])*6*3); // zero out
	alpha1[0][0] = 1.0; alpha1[3][1] = 1.0; alpha1[5][2] = 1.0;
	alpha2[1][1] = 1.0; alpha2[3][0] = 1.0; alpha2[4][2] = 1.0;
	alpha3[2][2] = 1.0; alpha3[4][1] = 1.0; alpha3[5][0] = 1.0;
	PetscScalar dNdxi[8]; PetscScalar dNdeta[8]; PetscScalar dNdzeta[8];
	PetscScalar J[3][3];
	PetscScalar invJ[3][3];
	PetscScalar beta[6][3];
	PetscScalar BMatrix[6][24]; // Note: Small enough to be allocated on stack
	PetscScalar *dN;
	// Make sure the stiffness matrix is zeroed out:
	memset(dke, 0, sizeof(dke[0])*24*24);
	// Perform the numerical integration
	for (PetscInt ii=0; ii<2-redInt; ii++){
		for (PetscInt jj=0; jj<2-redInt; jj++){
			for (PetscInt kk=0; kk<2-redInt; kk++){
				// Integration point
				PetscScalar xi = GP[ii];
				PetscScalar eta = GP[jj];
				PetscScalar zeta = GP[kk];
				// Differentiated shape functions
				DifferentiatedShapeFunctions(xi, eta, zeta, dNdxi, dNdeta, dNdzeta);
				// Jacobian
				J[0][0] = Dot(dNdxi,X,8); J[0][1] = Dot(dNdxi,Y,8); J[0][2] = Dot(dNdxi,Z,8);
				J[1][0] = Dot(dNdeta,X,8); J[1][1] = Dot(dNdeta,Y,8); J[1][2] = Dot(dNdeta,Z,8);
				J[2][0] = Dot(dNdzeta,X,8); J[2][1] = Dot(dNdzeta,Y,8); J[2][2] = Dot(dNdzeta,Z,8);
				// Inverse and determinant
				PetscScalar detJ = Inverse3M(J, invJ);
				// Weight factor at this point
				PetscScalar weight = W[ii]*W[jj]*W[kk]*detJ;
				// Strain-displacement matrix
				memset(BMatrix, 0, sizeof(BMatrix[0][0])*6*24); // zero out
				for (PetscInt ll=0; ll<3; ll++){
					// Add contributions from the different derivatives
					if (ll==0) {dN = dNdxi;}
					if (ll==1) {dN = dNdeta;}
					if (ll==2) {dN = dNdzeta;}
					// Assemble strain operator
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<3; j++){
							beta[i][j] = invJ[0][ll]*alpha1[i][j]
								+invJ[1][ll]*alpha2[i][j]
								+invJ[2][ll]*alpha3[i][j];
						}
					}
					// Add contributions to strain-displacement matrix
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<24; j++){
							BMatrix[i][j] = BMatrix[i][j] + beta[i][j%3]*dN[j/3];
						}
					}
				}
				// Finally, add to the element matrix
				for (PetscInt i=0; i<24; i++){
					for (PetscInt j=0; j<24; j++){
						for (PetscInt k=0; k<6; k++){
							for (PetscInt l=0; l<6; l++){

								dke[j+24*i] = dke[j+24*i] + weight*(BMatrix[k][i] * dC_drho[k][l] * BMatrix[l][j]);
							}
						}
					}
				}
			}
		}
	}

	return 0;
}

PetscScalar LinearElasticity::Dot(PetscScalar *v1, PetscScalar *v2, PetscInt l){
	// Function that returns the dot product of v1 and v2,
	// which must have the same length l
	PetscScalar result = 0.0;
	for (PetscInt i=0; i<l; i++){
		result = result + v1[i]*v2[i];
	}
	return result;
}

void LinearElasticity::DifferentiatedShapeFunctions(PetscScalar xi, PetscScalar eta, PetscScalar zeta, PetscScalar *dNdxi, PetscScalar *dNdeta, PetscScalar *dNdzeta){
	//differentiatedShapeFunctions - Computes differentiated shape functions
	// At the point given by (xi, eta, zeta).
	// With respect to xi:
	dNdxi[0]  = -0.125*(1.0-eta)*(1.0-zeta);
	dNdxi[1]  =  0.125*(1.0-eta)*(1.0-zeta);
	dNdxi[2]  =  0.125*(1.0+eta)*(1.0-zeta);
	dNdxi[3]  = -0.125*(1.0+eta)*(1.0-zeta);
	dNdxi[4]  = -0.125*(1.0-eta)*(1.0+zeta);
	dNdxi[5]  =  0.125*(1.0-eta)*(1.0+zeta);
	dNdxi[6]  =  0.125*(1.0+eta)*(1.0+zeta);
	dNdxi[7]  = -0.125*(1.0+eta)*(1.0+zeta);
	// With respect to eta:
	dNdeta[0] = -0.125*(1.0-xi)*(1.0-zeta);
	dNdeta[1] = -0.125*(1.0+xi)*(1.0-zeta);
	dNdeta[2] =  0.125*(1.0+xi)*(1.0-zeta);
	dNdeta[3] =  0.125*(1.0-xi)*(1.0-zeta);
	dNdeta[4] = -0.125*(1.0-xi)*(1.0+zeta);
	dNdeta[5] = -0.125*(1.0+xi)*(1.0+zeta);
	dNdeta[6] =  0.125*(1.0+xi)*(1.0+zeta);
	dNdeta[7] =  0.125*(1.0-xi)*(1.0+zeta);
	// With respect to zeta:
	dNdzeta[0]= -0.125*(1.0-xi)*(1.0-eta);
	dNdzeta[1]= -0.125*(1.0+xi)*(1.0-eta);
	dNdzeta[2]= -0.125*(1.0+xi)*(1.0+eta);
	dNdzeta[3]= -0.125*(1.0-xi)*(1.0+eta);
	dNdzeta[4]=  0.125*(1.0-xi)*(1.0-eta);
	dNdzeta[5]=  0.125*(1.0+xi)*(1.0-eta);
	dNdzeta[6]=  0.125*(1.0+xi)*(1.0+eta);
	dNdzeta[7]=  0.125*(1.0-xi)*(1.0+eta);
}

PetscScalar LinearElasticity::Inverse3M(PetscScalar J[][3], PetscScalar invJ[][3]){
	//inverse3M - Computes the inverse of a 3x3 matrix
	PetscScalar detJ = J[0][0]*(J[1][1]*J[2][2]-J[2][1]*J[1][2])-J[0][1]*(J[1][0]*J[2][2]-J[2][0]*J[1][2])+J[0][2]*(J[1][0]*J[2][1]-J[2][0]*J[1][1]);
	invJ[0][0] = (J[1][1]*J[2][2]-J[2][1]*J[1][2])/detJ;
	invJ[0][1] = -(J[0][1]*J[2][2]-J[0][2]*J[2][1])/detJ;
	invJ[0][2] = (J[0][1]*J[1][2]-J[0][2]*J[1][1])/detJ;
	invJ[1][0] = -(J[1][0]*J[2][2]-J[1][2]*J[2][0])/detJ;
	invJ[1][1] = (J[0][0]*J[2][2]-J[0][2]*J[2][0])/detJ;
	invJ[1][2] = -(J[0][0]*J[1][2]-J[0][2]*J[1][0])/detJ;
	invJ[2][0] = (J[1][0]*J[2][1]-J[1][1]*J[2][0])/detJ;
	invJ[2][1] = -(J[0][0]*J[2][1]-J[0][1]*J[2][0])/detJ;
	invJ[2][2] = (J[0][0]*J[1][1]-J[1][0]*J[0][1])/detJ;
	return detJ;
}

