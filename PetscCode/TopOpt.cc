#include <TopOpt.h>
#include <cmath>

TopOpt::TopOpt(){

  x=NULL;
  x1=NULL;
  x2=NULL;
  xPhys1=NULL;
  xPhys2=NULL;
  xDil1=NULL;
  xEro1=NULL;
  
  xvoid1=NULL;
  xsolid1=NULL;
  dfdx=NULL;
  dgdx=NULL;
  gx=NULL;
  
  xmin=NULL;
  xmax=NULL;
  XMIN=NULL;
  XMAX=NULL;
  xold=NULL;
  
  da_nodes=NULL;
  da_elem=NULL;
  da_elem2=NULL;
  
  xo1=NULL;
  xo2=NULL;
  U=NULL;
  L=NULL;
  
  dfdxSTG = NULL;

  SetUp();

}

TopOpt::~TopOpt(){

	if (x!=NULL){ VecDestroy(&x); }
    if (x1!=NULL){ VecDestroy(&x1); }
	if (x2!=NULL){ VecDestroy(&x2); }
	if (xPhys1!=NULL){ VecDestroy(&xPhys1); }
	if (xPhys2!=NULL){ VecDestroy(&xPhys2); }
	if (xDil1!=NULL){ VecDestroy(&xDil1); }
	if (xEro1!=NULL){ VecDestroy(&xEro1); }
	
	if (xvoid1!=NULL){ VecDestroy(&xvoid1); }
	if (xsolid1!=NULL){ VecDestroy(&xsolid1); }
	if (dfdx!=NULL){ VecDestroy(&dfdx); }
    if (dgdx!=NULL){ VecDestroyVecs(m,&dgdx); }
	if (gx!=NULL){ delete [] gx; }
	
    if (xold!=NULL){ VecDestroy(&xold); }
    if (xmin!=NULL){ VecDestroy(&xmin); }
    if (xmax!=NULL){ VecDestroy(&xmax); }
    if (XMIN!=NULL){ VecDestroy(&XMIN); }
    if (XMAX!=NULL){ VecDestroy(&XMAX); }
    
	if (da_nodes!=NULL){ DMDestroy(&(da_nodes)); }
    if (da_elem!=NULL){ DMDestroy(&(da_elem)); }
	if (da_elem2!=NULL){ DMDestroy(&(da_elem2)); }
	
    if (xo1!=NULL){ VecDestroy(&xo1); }
    if (xo2!=NULL){ VecDestroy(&xo2); }
    if (L!=NULL){ VecDestroy(&L); }
    if (U!=NULL){ VecDestroy(&U);  }

	delete [] weights;
	
	if (dfdxSTG!=NULL){ VecDestroy(&dfdxSTG); }
	
	for(int i=0;i<nslc;i++){
		VecDestroy(&xSTGload[i]);
		VecDestroy(&xSTGstiff1[i]);
		VecDestroy(&xSTGstiff2[i]);
		VecDestroy(&slcInputload[i]);
		VecDestroy(&slcInputstiff[i]);
	}
	
}

PetscErrorCode TopOpt::SetUp(){
	
	PetscErrorCode ierr;

	// Set hardcoded data for optimization problems
	maxItr = 120;
    penal = 1.0;
    filter = 2; // 0=sens,1=dens,2=PDE - other val == no filtering
    m = 2; // Two volume constraints
    Xmin = 0.0;
    Xmax = 1.0;
    movlim = 0.2;
	
	// Read inputs from files
	// First the data from voxelization	
	// Start reading the file
	{
		std::string fname="../input/optimization_data.txt";
		std::string tmpstr;
		std::fstream instr;
		instr.open(fname.c_str(),std::ios::in);
		while (std::getline(instr, tmpstr)){
			instr>>tmpstr;
			std::transform(tmpstr.begin(), tmpstr.end(), tmpstr.begin(), ::tolower);
			if(tmpstr=="nbc"){
				instr>>nbc;			
			}
			else if (tmpstr=="nlc"){
				tnlc = 0;
				for (int i=0; i<nbc; i++){
					instr>>nlc[i];
					tnlc = tnlc + nlc[i];
				}
			}
			else if (tmpstr=="weights"){
				// Allocate after input
				weights = new PetscScalar[tnlc];
				for (int i=0; i<tnlc; i++){
					instr>>weights[i];
				}
			} 			
			else if(tmpstr=="bb"){
				instr>>xc[0];
				instr>>xc[1];
				instr>>xc[2];
				instr>>xc[3];
				instr>>xc[4];
				instr>>xc[5];
			}
			else if(tmpstr=="dc"){
				instr>>nxyz[0];
				instr>>nxyz[1];
				instr>>nxyz[2];
				instr>>nlvls;
			}
			else if(tmpstr=="nslc"){
				instr>>nslc;
			}
			else if(tmpstr=="printdir"){
				instr>>printdir;
			}
			else if(tmpstr=="printweight"){
				instr>>printweight;
			}
			else if(tmpstr=="interpolation"){
				instr>>interprule;
			}
			else {
				continue;
			}
		}
		instr.close();
	}
	// End reading the file 
	// Second comes the input from user (volfrac only in this version)	
	
	// Start reading the file
	{
		std::string fname="../input/optimization_input.txt";
		std::string tmpstr;
		std::fstream instr;
		instr.open(fname.c_str(),std::ios::in);
		while (std::getline(instr, tmpstr)){
			instr>>tmpstr;
			std::transform(tmpstr.begin(), tmpstr.end(), tmpstr.begin(), ::tolower);
			if(tmpstr=="volfrac"){
				instr>>volfracin1;
				instr>>volfracin2;
			}
			else if(tmpstr=="matprop"){
				instr>>Emax;
				instr>>nu;
				instr>>Sy;
				instr>>density;
			}
			else {
				continue;
			}
		}
		instr.close();
	}
	// End reading the file 
	Emin = Emax/1.0e6; // 6 orders difference
	
	ierr = SetUpMESH(); CHKERRQ(ierr);

	ierr = SetUpOPT(); CHKERRQ(ierr);

	return(ierr);
}

PetscErrorCode TopOpt::SetUpMESH(){

	PetscErrorCode ierr;

	// Write parameters for the physics 
	PetscPrintf(PETSC_COMM_WORLD,"########################################################################\n");
	PetscPrintf(PETSC_COMM_WORLD,"############################ FEM settings ##############################\n");
	PetscPrintf(PETSC_COMM_WORLD,"# Number of nodes: (-nx,-ny,-nz):        (%i,%i,%i) \n",nxyz[0],nxyz[1],nxyz[2]);
    PetscPrintf(PETSC_COMM_WORLD,"# Number of degree of freedom:           %i \n",3*nxyz[0]*nxyz[1]*nxyz[2]);
	PetscPrintf(PETSC_COMM_WORLD,"# Number of elements:                    (%i,%i,%i) \n",nxyz[0]-1,nxyz[1]-1,nxyz[2]-1);
	PetscPrintf(PETSC_COMM_WORLD,"# Dimensions: (-xcmin,-xcmax,..,-zcmax): (%f,%f,%f)\n",xc[1]-xc[0],xc[3]-xc[2],xc[5]-xc[4]);
	PetscPrintf(PETSC_COMM_WORLD,"# -nlvls: %i\n",nlvls);
	PetscPrintf(PETSC_COMM_WORLD,"# -nbc: %i\n",nbc);
	for (int i=0; i<nbc; i++){
		PetscPrintf(PETSC_COMM_WORLD,"# -nlc in BC %i: %i\n",i+1,nlc[i]);
	}
	PetscPrintf(PETSC_COMM_WORLD,"# -tnlc: %i\n",tnlc);
    PetscPrintf(PETSC_COMM_WORLD,"########################################################################\n");

	// Check if the mesh supports the chosen number of MG levels
	PetscScalar divisor = PetscPowScalar(2.0,(PetscScalar)nlvls-1.0);
	// x - dir
	if ( std::floor((PetscScalar)(nxyz[0]-1)/divisor) != (nxyz[0]-1.0)/((PetscInt)divisor) ) {
		PetscPrintf(PETSC_COMM_WORLD,"MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD,"X - number of nodes %i is cannot be halfened %i times\n",nxyz[0],nlvls-1);
		exit(0);
	}
	// y - dir
    if ( std::floor((PetscScalar)(nxyz[1]-1)/divisor) != (nxyz[1]-1.0)/((PetscInt)divisor) ) {
        PetscPrintf(PETSC_COMM_WORLD,"MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD,"Y - number of nodes %i is cannot be halfened %i times\n",nxyz[1],nlvls-1);
		exit(0);
    }
	// z - dir
    if ( std::floor((PetscScalar)(nxyz[2]-1)/divisor) != (nxyz[2]-1.0)/((PetscInt)divisor) ) {
        PetscPrintf(PETSC_COMM_WORLD,"MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD,"Z - number of nodes %i is cannot be halfened %i times\n",nxyz[2],nlvls-1);
		exit(0);
    }

	// Start setting up the FE problem
	// Boundary types: DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_PERIODIC
	DMBoundaryType bx = DM_BOUNDARY_NONE;
	DMBoundaryType by = DM_BOUNDARY_NONE;
	DMBoundaryType bz = DM_BOUNDARY_NONE;

	// Stencil type - box since this is closest to FEM (i.e. STAR is FV/FD)
	DMDAStencilType  stype = DMDA_STENCIL_BOX;

	// Discretization: nodes:
	// For standard FE - number must be odd
	// For periodic: Number must be even
	PetscInt nx = nxyz[0];
	PetscInt ny = nxyz[1];
	PetscInt nz = nxyz[2];

	// number of nodal dofs
	PetscInt numnodaldof = 3;

	// Stencil width: each node connects to a box around it - linear elements
	PetscInt stencilwidth = 1;

	// Coordinates and element sizes
	PetscReal xmin=xc[0], xmax=xc[1], ymin=xc[2], ymax=xc[3], zmin=xc[4], zmax=xc[5];
	dx = (xc[1]-xc[0])/(PetscScalar(nxyz[0]-1));
	dy = (xc[3]-xc[2])/(PetscScalar(nxyz[1]-1));
	dz = (xc[5]-xc[4])/(PetscScalar(nxyz[2]-1));
	
	// Set the filter radius
	PetscReal dmax = 3.0*PetscMax(dx,PetscMax(dy,dz));
	rmin = PetscMax(dmax,1.0);
	
	// Create the nodal mesh
	ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stype,nx,ny,nz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
			numnodaldof,stencilwidth,0,0,0,&(da_nodes));
	CHKERRQ(ierr);

	// Set the coordinates
	ierr = DMDASetUniformCoordinates(da_nodes, xmin,xmax, ymin,ymax, zmin,zmax);
	CHKERRQ(ierr);

	// Set the element type to Q1: Otherwise calls to GetElements will change to P1 !
	ierr = DMDASetElementType(da_nodes, DMDA_ELEMENT_Q1);
	CHKERRQ(ierr);

	// Create the element mesh
	// find the geometric partitioning of the nodal mesh, so the element mesh will coincide
	// with the nodal mesh
	PetscInt md,nd,pd;
	ierr = DMDAGetInfo(da_nodes,NULL,NULL,NULL,NULL,&md,&nd,&pd,NULL,NULL,NULL,NULL,NULL,NULL);
	CHKERRQ(ierr);

	// vectors with partitioning information
	PetscInt *Lx=new PetscInt[md];
	PetscInt *Ly=new PetscInt[nd];
	PetscInt *Lz=new PetscInt[pd];

	// get number of nodes for each partition
	const PetscInt *LxCorrect, *LyCorrect, *LzCorrect;
	ierr = DMDAGetOwnershipRanges(da_nodes, &LxCorrect, &LyCorrect, &LzCorrect);
	CHKERRQ(ierr);

	// subtract one from the lower left corner.
	for (int i=0; i<md; i++){
		Lx[i] = LxCorrect[i];
		if (i==0){Lx[i] = Lx[i]-1;}
	}
	for (int i=0; i<nd; i++){
		Ly[i] = LyCorrect[i];
		if (i==0){Ly[i] = Ly[i]-1;}
	}
	for (int i=0; i<pd; i++){
		Lz[i] = LzCorrect[i];
		if (i==0){Lz[i] = Lz[i]-1;}
	}

	// Create the element grid: NOTE CONNECTIVITY IS 0
	PetscInt conn = 0;
	ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stype,nx-1,ny-1,nz-1,md,nd,pd,
			1,conn,Lx,Ly,Lz,&(da_elem));
	CHKERRQ(ierr);
	
	// Create the element grid with 2 dof per node
	ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stype,nx-1,ny-1,nz-1,md,nd,pd,
			2,conn,Lx,Ly,Lz,&(da_elem2));
	CHKERRQ(ierr);

	// Set element center coordinates
	ierr = DMDASetUniformCoordinates(da_elem , xmin+dx/2.0,xmax-dx/2.0, ymin+dy/2.0,ymax-dy/2.0, zmin+dz/2.0,zmax-dz/2.0);
	CHKERRQ(ierr);
	ierr = DMDASetUniformCoordinates(da_elem2, xmin+dx/2.0,xmax-dx/2.0, ymin+dy/2.0,ymax-dy/2.0, zmin+dz/2.0,zmax-dz/2.0);
	CHKERRQ(ierr);

	// Clean up
	delete [] Lx;
	delete [] Ly;
	delete [] Lz;

  	return(ierr);
}

PetscErrorCode TopOpt::SetUpOPT(){

	PetscErrorCode ierr;

	ierr = DMCreateGlobalVector(da_elem,&x1);  CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(da_elem2,&x);  CHKERRQ(ierr);

	// Total number of design variables
	VecGetSize(x,&n);
	// Total number of voxels 
	VecGetSize(x1,&nvoxel);
	
	PetscBool flg;

	// Optimization parameters - only maxItr and rmin for tests 
	PetscOptionsGetReal(NULL,NULL,"-rmin",&rmin,&flg);
	PetscOptionsGetInt(NULL,NULL,"-maxItr",&maxItr,&flg);

    // Allocate after input
    gx = new PetscScalar[m];
	if (filter==0){
		Xmin = 0.001; // Prevent division by zero in filter
	}
    
	// Read in the voids and solids vectors
    Vec xvoid, xsolid;
    MPI_Comm comm;
    PetscViewer viewer1, viewer2;
    comm = PETSC_COMM_WORLD;
	ierr = DMDACreateNaturalVector(da_elem,&xvoid); CHKERRQ(ierr); CHKERRQ(ierr); // Read in natural ordering
	ierr = DMDACreateNaturalVector(da_elem,&xsolid); CHKERRQ(ierr); CHKERRQ(ierr); // Read in natural ordering
	ierr = DMCreateGlobalVector(da_elem,&xvoid1); CHKERRQ(ierr); CHKERRQ(ierr); // Copy to DMDA ordering
    ierr = DMCreateGlobalVector(da_elem,&xsolid1); CHKERRQ(ierr); CHKERRQ(ierr); // Copy to DMDA ordering
    ierr = PetscViewerBinaryOpen(comm,"../input/void.bin",FILE_MODE_READ,&viewer1); CHKERRQ(ierr); // Read binary file with voids data
    ierr = VecLoad(xvoid,viewer1); CHKERRQ(ierr); // Keep void data in new natural vector
    ierr = PetscViewerBinaryOpen(comm,"../input/solid.bin",FILE_MODE_READ,&viewer2); CHKERRQ(ierr); // Read binary file with solids data
    ierr = VecLoad(xsolid,viewer2); CHKERRQ(ierr); // Keep solid data in new natural vector
    ierr = PetscViewerDestroy(&viewer1); CHKERRQ(ierr); // Destroy viewer
    ierr = PetscViewerDestroy(&viewer2); CHKERRQ(ierr); // Destroy viewer
    ierr = DMDANaturalToGlobalBegin(da_elem,xvoid,INSERT_VALUES,xvoid1); CHKERRQ(ierr); // Transfer vector to global ordering - start
    ierr = DMDANaturalToGlobalEnd(da_elem,xvoid,INSERT_VALUES,xvoid1); CHKERRQ(ierr); // Transfer vector to global ordering - end
    ierr = DMDANaturalToGlobalBegin(da_elem,xsolid,INSERT_VALUES,xsolid1); CHKERRQ(ierr); // Transfer vector to global ordering - start
    ierr = DMDANaturalToGlobalEnd(da_elem,xsolid,INSERT_VALUES,xsolid1); CHKERRQ(ierr); // Transfer vector to global ordering - end
	VecDestroy(&xvoid);
	VecDestroy(&xsolid);
	
	// // View solids and voids - for debugging
	// {
		// PetscViewer viewerVTS;
		// ierr = PetscViewerVTKOpen(comm,"../output/solid_check.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		// ierr = VecView(xsolid1,viewerVTS); CHKERRQ(ierr); // View vector
		// ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
		
		// ierr = PetscViewerVTKOpen(comm,"../output/void_check.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		// ierr = VecView(xvoid1,viewerVTS); CHKERRQ(ierr); // View vector
		// ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
		
	// }
	// PetscScalar checksolidvoid;
	// ierr = VecDot(xsolid1,xvoid1,&checksolidvoid);
	// PetscPrintf(PETSC_COMM_WORLD,"Dot product of solids and voids is %f \n",checksolidvoid);
	
	// Compute actual volume fraction from total box
	PetscScalar sumxvoid;
	ierr = VecSum(xvoid1,&sumxvoid);
	volfrac1 = volfracin1*(nvoxel - sumxvoid)/nvoxel; // Actual volume fraction for solid/void material
	volfrac2 = volfracin2; // Actual volume fraction for lattice 
	
	// Allocate all density vectors 
	ierr = VecDuplicate(x1,&x2); CHKERRQ(ierr);
	ierr = VecDuplicate(x1,&xTilde1); CHKERRQ(ierr);
	ierr = VecDuplicate(x1,&xEro1); CHKERRQ(ierr);
	ierr = VecDuplicate(x1,&xPhys1); CHKERRQ(ierr);
	ierr = VecDuplicate(x1,&xDil1); CHKERRQ(ierr);
	ierr = VecDuplicate(x2,&xTilde2); CHKERRQ(ierr);
	ierr = VecDuplicate(x2,&xPhys2); CHKERRQ(ierr);
	
	// Sensitivity vectors
	ierr = VecDuplicate(x,&dfdx); CHKERRQ(ierr);
	ierr = VecDuplicateVecs(x,m,&dgdx); CHKERRQ(ierr);
	
	// Set initial vectors
	ierr = VecStrideSet(x,0,volfrac1); // Set the first variable to volfrac1
	PetscScalar volfracinit = 0.8*volfrac2;
	ierr = VecStrideSet(x,1,volfracinit); // Set the second variable to volfrac2
	
	// Consider solids and voids in initial design
    ierr = VecScale(xvoid1,-1.0); // Compute -xvoid
    ierr = VecShift(xvoid1,+1.0); // Compute 1-xvoid
	Vec dummyvec; // Create dummy vector with one variable per node
	ierr = VecDuplicate(x1,&dummyvec);
	ierr = VecStrideGather(x,0,dummyvec,INSERT_VALUES); // x1 variables inserted to dummyvec
    ierr = VecPointwiseMax(dummyvec,xsolid1,dummyvec); // Turn solids into 1's
	ierr = VecPointwiseMin(dummyvec,xvoid1,dummyvec); // Turn voids into 0's
	ierr = VecStrideScatter(dummyvec,0,x,INSERT_VALUES); // Insert back into x
	
	// Bounds and old vecs
	VecDuplicate(x,&xmin);
	VecDuplicate(x,&xmax);
	VecDuplicate(x,&xold);
	VecCopy(x,xold);

	// Global bounds for design variables 
	VecDuplicate(x,&XMIN);
	VecDuplicate(x,&XMAX);
	VecSet(XMIN,0.0);
	VecStrideSet(XMAX,0,1.0); // Solid/void can reach 1.0
	VecStrideSet(XMAX,1,0.5); // Lattice cannot exceed 0.5
	
	// Turn lower bound of solids into 0.999 for x1
	ierr = VecScale(xsolid1,0.999); // Compute solid*0.999
	ierr = VecStrideGather(XMIN,0,dummyvec,INSERT_VALUES); // XMIN for existence variables inserted to dummyvec
	ierr = VecPointwiseMax(dummyvec,xsolid1,dummyvec);
	ierr = VecStrideScatter(dummyvec,0,XMIN,INSERT_VALUES); // Insert back into XMIN
	ierr = VecScale(xsolid1,1.0009); // Set xsolid1 back to (almost) 1
	
	// Turn upper bound of solids into 0.001 for x2
    ierr = VecStrideGather(XMAX,1,dummyvec,INSERT_VALUES); // XMAX for material variables inserted to dummyvec
	ierr = VecAXPY(dummyvec,-0.499,xsolid1);
	ierr = VecStrideScatter(dummyvec,1,XMAX,INSERT_VALUES); // Insert back into XMAX

	// Turn upper bound of voids into 0.001
    ierr = VecShift(xvoid1,-1.0); // Compute -xvoid
    ierr = VecScale(xvoid1,-1.0); // Compute xvoid
	ierr = VecStrideGather(XMAX,0,dummyvec,INSERT_VALUES); // XMAX for existence variables inserted to dummyvec
    ierr = VecAXPY(dummyvec,-0.999,xvoid1);
	ierr = VecStrideScatter(dummyvec,0,XMAX,INSERT_VALUES); // Insert back into XMAX
	ierr = VecStrideGather(XMAX,1,dummyvec,INSERT_VALUES); // XMAX for material variables inserted to dummyvec
    ierr = VecAXPY(dummyvec,-0.499,xvoid1);
	// ierr = VecPointwiseMax(dummyvec,0.001,dummyvec); // Ensure no values below zero
	ierr = VecStrideScatter(dummyvec,1,XMAX,INSERT_VALUES); // Insert back into XMAX
	
	ierr = VecDestroy(&dummyvec);
	
	PetscPrintf(PETSC_COMM_WORLD,"################### Optimization settings ####################\n");
	PetscPrintf(PETSC_COMM_WORLD,"# Problem size: n= %i, m= %i\n",n,m);
	PetscPrintf(PETSC_COMM_WORLD,"# -filter: %i  (0=sens., 1=dens, 2=PDE)\n",filter);
	PetscPrintf(PETSC_COMM_WORLD,"# -rmin: %f\n",rmin);
	PetscPrintf(PETSC_COMM_WORLD,"# -volfrac1: %f\n",volfrac1);
	PetscPrintf(PETSC_COMM_WORLD,"# -volfrac2: %f\n",volfrac2);
    PetscPrintf(PETSC_COMM_WORLD,"# -penal: %f\n",penal);
	PetscPrintf(PETSC_COMM_WORLD,"# -Emin/-Emax: %e - %e \n",Emin,Emax);
	PetscPrintf(PETSC_COMM_WORLD,"# -maxItr: %i\n",maxItr);
	PetscPrintf(PETSC_COMM_WORLD,"# -movlim: %f\n",movlim);
    PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");
	
	/////////////////////////////////
	// *** STAGED CONSTRUCTION *** //
	/////////////////////////////////
	char inputfilename [PETSC_MAX_PATH_LEN];
	int nfilename;
	Vec inputvec;
	
	ierr = DMDACreateNaturalVector(da_elem,&inputvec); CHKERRQ(ierr); CHKERRQ(ierr); // Read in natural ordering
	
	// Set initial xSTG and slcInput (loop over total number of slices)
	xSTGload.resize(nslc);
	xSTGstiff1.resize(nslc);
	xSTGstiff2.resize(nslc);
	slcInputload.resize(nslc);
	slcInputstiff.resize(nslc);
	
	for(PetscInt i=0; i<nslc; i++){
		ierr = DMCreateGlobalVector(da_elem,&(xSTGload[i])); CHKERRQ(ierr);
		VecSet(xSTGload[i],0.0);
		ierr = DMCreateGlobalVector(da_elem,&(xSTGstiff1[i])); CHKERRQ(ierr);
		VecSet(xSTGstiff1[i],0.0);
		ierr = DMCreateGlobalVector(da_elem,&(xSTGstiff2[i])); CHKERRQ(ierr);
		VecSet(xSTGstiff2[i],0.0);
		ierr = DMCreateGlobalVector(da_elem,&(slcInputload[i])); CHKERRQ(ierr);
		VecSet(slcInputload[i],0.0);
		ierr = DMCreateGlobalVector(da_elem,&(slcInputstiff[i])); CHKERRQ(ierr);
		VecSet(slcInputstiff[i],0.0);
	}
	
	// Allocate sensitivity vector
	ierr = VecDuplicate(x,&dfdxSTG); CHKERRQ(ierr);
	
	// Read slicing data and insert into slcInputload
	for(PetscInt i=0; i<nslc; i++){
		nfilename = sprintf(inputfilename,"../input/SLC_%i.bin",i+1);
		ierr = PetscViewerBinaryOpen(comm,inputfilename,FILE_MODE_READ,&viewer1); CHKERRQ(ierr); // Read binary file with slice data
		ierr = VecLoad(inputvec,viewer1); CHKERRQ(ierr); // Keep slice data in new natural vector
		ierr = PetscViewerDestroy(&viewer1); CHKERRQ(ierr); // Destroy viewer
		ierr = DMDANaturalToGlobalBegin(da_elem,inputvec,INSERT_VALUES,slcInputload[i]); CHKERRQ(ierr); // Transfer vector to global ordering - start
		ierr = DMDANaturalToGlobalEnd(da_elem,inputvec,INSERT_VALUES,slcInputload[i]); CHKERRQ(ierr); // Transfer vector to global ordering - end
	}
	
	// Create sum of slicing data for stiffness
	ierr = VecCopy(slcInputload[0],slcInputstiff[0]);
	for(PetscInt i=1; i<nslc; i++){
		ierr = VecCopy(slcInputload[i],slcInputstiff[i]);
		ierr = VecAXPY(slcInputstiff[i],1.0,slcInputstiff[i-1]);
	}
	
	// {
		// PetscViewer viewerVTS;
		// ierr = PetscViewerVTKOpen(comm,"../output/slc_check.vts",FILE_MODE_WRITE,&viewerVTS); CHKERRQ(ierr); // Open viewer
		// ierr = VecView(slcInput[0],viewerVTS); CHKERRQ(ierr); // View vector
		// ierr = PetscViewerDestroy(&viewerVTS); CHKERRQ(ierr); // Close viewer
	// }
	
	VecDestroy(&inputvec);
	
	return(ierr);
}

void TopOpt::AllocMMAwithRestart(int *itr, MMA **mma)  {

  	// Check if restart is desired
	restart = PETSC_FALSE; // DEFAULT DOES NOT USE RESTART
	flip = PETSC_TRUE;     // BOOL to ensure that two dump streams are kept

	PetscBool flg;
	PetscOptionsGetBool(NULL,NULL,"-restart",&restart,&flg);

	// Where to put restart files
	char filenameChar[PETSC_MAX_PATH_LEN];
    PetscOptionsGetString(NULL,NULL,"-workdir",filenameChar,sizeof(filenameChar),&flg);
    std::string filenameWorkdir = "./";
    if (flg){
        filenameWorkdir = "";
        filenameWorkdir.append(filenameChar);
    }

    std::string filename;
    // Check PETSc input for a data directory (to see if we should load restart files from somewhere else)
    PetscOptionsGetString(NULL,NULL,"-restartDir",filenameChar,sizeof(filenameChar),&flg);
    // If input, change path of the file in filename
    if (flg){
        filename="";
        filename.append(filenameChar);
    }
    else {
        filename = filenameWorkdir;
    }


	// Which solution to use for restarting
	PetscInt restartNumber = 1;
	PetscOptionsGetInt(NULL,NULL,"-restartNumber",&restartNumber,&flg);

	PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");
	PetscPrintf(PETSC_COMM_WORLD,"# Continue from previous iteration (-restart): %i \n",restart);
	PetscPrintf(PETSC_COMM_WORLD,"# Restart files are located in folder (-restartDir): %s \n",filename.c_str());
	PetscPrintf(PETSC_COMM_WORLD,"# New restart files are written to folder (-workdir): %s \n",filenameWorkdir.c_str());

	// Append the dummyname for restart files
	filename.append("/restore_V");
	filenameWorkdir.append("/restore_V");

	PetscPrintf(PETSC_COMM_WORLD,"# The restart point is restore_V%i****.dat  (where %i is the -restartNumber) \n",restartNumber,restartNumber);

	// RESTORE FROM BREAKDOWN
	PetscInt myrank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
    std::ifstream indes;
	// Name of write files
    restdens_1 = filenameWorkdir;
    restdens_2 = filenameWorkdir;
    restdens_1.append("1");
    restdens_2.append("2");

	// Name of read files
	std::string restdens_1_read=filename;
    restdens_1_read.append("1");
    std::string restdens_2_read=filename;
    restdens_2_read.append("2");

	std::string zerosString;
	std::stringstream ss;
	if(myrank<10){
        zerosString = "_0000";
    }
    else if(myrank<100){
        zerosString = "_000";
    }
    else if(myrank<1000){
        zerosString = "_00";
    }
    ss << restdens_1 << zerosString << myrank << ".dat";
    ss << " "; // Space to separate filenames
    ss << restdens_2 << zerosString << myrank << ".dat";
    ss << " "; // Space to separate filenames
    ss << restdens_1_read << zerosString << myrank << ".dat";
    ss << " "; // Space to separate filenames
    ss << restdens_2_read << zerosString << myrank << ".dat";
    // Put file names back into strings
    ss >> restdens_1;
    ss >> restdens_2;
    ss >> restdens_1_read;
    ss >> restdens_2_read;

	// Allocate the data needed for a MMA restart
	VecDuplicate(x,&xo1);
	VecDuplicate(x,&xo2);
	VecDuplicate(x,&U);
	VecDuplicate(x,&L);

	// Read from restart point
	if (restartNumber==1){
		indes.open(restdens_1_read.c_str(),std::ios::in);
	}
	else if (restartNumber==2){
		indes.open(restdens_2_read.c_str(),std::ios::in);
	}

	if(indes && restart)
	{
		PetscInt nlocsiz;
		PetscScalar *xp, *xpp, *xo1p, *xo2p, *Up, *Lp;

		VecGetArray(x,&xp);
		VecGetArray(xPhys1,&xpp);

		VecGetArray(xo1,&xo1p);
		VecGetArray(xo2,&xo2p);
		VecGetArray(U,&Up);
		VecGetArray(L,&Lp);

		indes.read((char*)&nlocsiz,sizeof(PetscInt));
		indes.read((char*)xp,sizeof(PetscScalar)*nlocsiz);
		indes.read((char*)xpp,sizeof(PetscScalar)*nlocsiz);
		indes.read((char*)xo1p,sizeof(PetscScalar)*nlocsiz);
		indes.read((char*)xo2p,sizeof(PetscScalar)*nlocsiz);
		indes.read((char*)Up,sizeof(PetscScalar)*nlocsiz);
		indes.read((char*)Lp,sizeof(PetscScalar)*nlocsiz);
		indes.read((char*)itr,sizeof(PetscInt));
		indes.read((char*)&fscale,sizeof(PetscScalar));
		indes.close();

		VecRestoreArray(x,&xp);
		VecRestoreArray(xPhys1,&xpp);
		VecRestoreArray(xo1,&xo1p);
		VecRestoreArray(xo2,&xo2p);
		VecRestoreArray(U,&Up);
		VecRestoreArray(L,&Lp);

		*mma = new MMA(n,m,*itr,xo1,xo2,U,L);

		if (restartNumber==1){
			PetscPrintf(PETSC_COMM_WORLD,"# Successful restart from file (starting from): %s \n",restdens_1_read.c_str());
		}
		else if (restartNumber==2){
			PetscPrintf(PETSC_COMM_WORLD,"# Successful restart from file (starting from): %s \n",restdens_2_read.c_str());
		}


	}
	else {
		*mma = new MMA(n,m,x);
	}

	indes.close();

}

void TopOpt::WriteRestartFiles(int *itr, MMA *mma) {

	// Always dump data if correct allocater has been used
	if (xo1!=NULL){
		// Get data from MMA
		mma->Restart(xo1,xo2,U,L);

		// Open the stream (and make sure there always is one working copy)
		std::string dens_iter;
		std::stringstream ss_iter;
		if (flip) {
			ss_iter << restdens_1;
			ss_iter >> dens_iter;
			flip = PETSC_FALSE;
		} else {
			ss_iter << restdens_2;
			ss_iter >> dens_iter;
			flip = PETSC_TRUE;
		}

		// Open stream
		std::ofstream out(dens_iter.c_str(),std::ios::out);

		// poniters to data
		PetscInt nlocsiz;
		VecGetLocalSize(x,&nlocsiz);
		PetscScalar *xp, *xpp, *xo1p, *xo2p, *Up, *Lp;

		VecGetArray(x,&xp);
		VecGetArray(xPhys1,&xpp);
		VecGetArray(xo1,&xo1p);
		VecGetArray(xo2,&xo2p);
		VecGetArray(U,&Up);
		VecGetArray(L,&Lp);

		// Write to file
		out.write((char*)&nlocsiz,sizeof(PetscInt));
		out.write((char*)xp,sizeof(PetscScalar)*nlocsiz);
		out.write((char*)xpp,sizeof(PetscScalar)*nlocsiz);
		out.write((char*)xo1p,sizeof(PetscScalar)*nlocsiz);
		out.write((char*)xo2p,sizeof(PetscScalar)*nlocsiz);
		out.write((char*)Up,sizeof(PetscScalar)*nlocsiz);
		out.write((char*)Lp,sizeof(PetscScalar)*nlocsiz);
		out.write((char*)itr,sizeof(PetscInt));
		out.write((char*)&fscale,sizeof(PetscScalar));
		out.close();

		// Tidy up
		VecRestoreArray(x,&xp);
		VecRestoreArray(xPhys1,&xpp);
		VecRestoreArray(xo1,&xo1p);
		VecRestoreArray(xo2,&xo2p);
		VecRestoreArray(U,&Up);
		VecRestoreArray(L,&Lp);
	}


}
