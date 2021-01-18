# concurrent-structure-support-4AM
PETSc-based ocde used for concurrent topopt of structure and supports for AM

Notes on the code for concurrent optimization of structure and its support
--------------------------------------------------------------------------
The provided code represents the case of two material phases for the support corresponding to row #3 in Table #8 in the article:
"Concurrent high-resolution topology optimization of structures and their supports for additive manufacturing"
http://dx.doi.org/10.1007/s00158-020-02835-6
The code is based on the original PETSc code for large-scale topology optimization, described in the article:
"Topology optimization using PETSc: An easy-to-use, fully parallel, open source topology optimization framework"
https://link.springer.com/article/10.1007/s00158-014-1157-0
See also github repository:
https://github.com/topopt/TopOpt_in_PETSc

Content of the folders:

input (contains inputs needed to execute an optimization)
-----
1. BC_1.bin:  Definition of the boundary conditions. In the provided example, a subset of elements attached to the fixed wall at z = zmax.

2. BCstg.bin: Definition of the boundary conditions for the layered construction response, a subset of elements at the bottom side of the printing direction (the printing bed). In the provided example, at y = 0 to y = 4.

3. solid.bin: A solid non-design subset of elements that carries the external load. In the provided example: 3x3 elements in the +y and +z directions along the x direction at z = y = 0. 

4. void.bin:  The physical design space without the subsets of the load carrying elements, the structural BCs elements and the BCs elements that function as printing bed.

5. SLC_1.bin, SLC_2.bin, SLC_3.bin: The subdomains of the slices of the layered construction response. In the provided example: three equally spaced layers in the y direction.

6. RHS_1_1_1.bin: The right hand side of the static equation for the external load.

7. optimization_data.txt: Data file containing parameters needed for running the code. For details see comments inside the file starting with the # sign.

8. optimization_input.txt: Complimentary data required to run the code. For details see comments inside the file starting with the # sign. Important! The 1st row defines the fraction of the two material phases. In the provided example: vf1 = 0.2 is the volume fraction of the solid phase, vf2 = 0.02 is the volume fraction of the lattice phase.

output
------
The outputs of the optimization are stored in this folder. In the provided example, it contains two vts files of the two material phases. It is advised to leave this folder empty before running the program.

PetscCode
---------
This folder contains all required PETCs files to run the program. These files are based on extending the parallel topology optimization code that utilizes the PETSc library, see: N. Aage, E. Andreassen, and B. S. Lazarov. "Topology optimization using petsc: An easy-to-use, fully parallel, open source topology optimization framework." Structural and Multidisciplinary Optimization, 51(3): 565-572, 2015. https://link.springer.com/article/10.1007/s00158-014-1157-0
See also github repository:
https://github.com/topopt/TopOpt_in_PETSc

To run the code on a Linux based system you should compile it first, then execute the run with a proper shell script. The 'input' and 'output' folders should be placed in the same level as the 'PetscCode' folder.
The 'input' folder should contain all the above mentioned input files.

voxelize_preproc.m
------------------
A Matlab script that produces all the required files for the input folder. It utilizes the mesh voxelization package, https://www.mathworks.com/matlabcentral/fileexchange/27390-mesh-voxelisation
It is advised to follow the detailed comments in the code.

The script is divided into 7 sections detailed herein:
------------------------------------------------------
1. rows 1-4:   Initialization and definition of printing direction.
2. rows 5-19:  Determine the design domain. 
3. rows 20-27: Voxelization of the design domain.
4. rows 28-47: Creation of slices for the layered construction response.
5. rows 48-59: Definition and creation of the solid region that carries the external load.
6. rows 60-81: Definition and creation of boundary conditions for the final situation (the full structure) and for the layered construction response.
7. rows 82-94: Definition of the external load cases and determination of the static equation right hand side.


