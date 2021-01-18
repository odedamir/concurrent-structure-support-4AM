#ifndef __MMA__
#define __MMA__

#include <petsc.h>

/* -----------------------------------------------------------------------------
MMA Copyright (C) 2013-2014,
This MMA implementation is licensed under Version 2.1 of the GNU
Lesser General Public License.

This MMA implementation is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This Module is distributed in the hope that it will be useful,implementation
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
-------------------------------------------------------------------------- */

/*
Please cite the following usage of this class:

1) Svanberg K (1987) The method of moving asymptotes - a new
method for structural optimization. International
Journal for Numerical Methods in Engineering 25
 - and -
2) Aage N, Andreassen E and Lazarov B, Topology optimization using PETSc:
An easy-to-use, fully parallel, open source topology optimization framework,
submitted 2014

*/



/*
Implementation by Niels Aage, August 2013.

PETSc implementation of the Method of Moving Asymptotes

The class solves a general non-linear programming problem
on standard from, i.e. non-linear objective f, m non-linear
inequality constraints g and box constraints on the n
design variables xmin, xmax.

       min_x^n f(x)
       s.t. g_j(x) < 0,   j = 1,m
       xmin < x_i < xmax, i = 1,n

Each call to Update() sets up and solve the following
convex subproblem:

  min_x     sum(p0j./(U-x)+q0j./(x-L)) + z + 0.5*z^2 + sum(c.*y + 0.5*d.*y.^2)

  s.t.      sum(pij./(U-x)+qij./(x-L)) - ai*z - yi <= bi, i = 1,m
            Lj < alphaj <=  xj <= betaj < Uj,  j = 1,n
            yi >= 0, i = 1,m
            z >= 0.

The subproblem is solved using the dual formulation and
a primal/dual interior point method.

----------------------------------------------------------------------
// Example usage:

// ALLOCATE:
MMA *mma = new MMA(n,m,x);  // where x is a PETSc Vec (Seq/MPI)
// OR //
MMA *mma = new MMA(n,m,x,a,c,d); // a,c,d can be set afterwards using
				 // SetAsymptotes(...)

// IF needed:
SetAsymptotes(..)		// Set the initial, increase and decrease factors.
ConstraintModification(..)
SetRobustAsymptotesType(...)

while (not converged){

	// compute objective, constraints and sens

	// IF needed: rescale the bounds
	mma->SetOuterMovelimi(...)

	// Update the design field
	mma->Update(...)

	// IF needed: get residual of KKT sys
	mma->KKTresidual(...)
	// OR get inf_norm of the design change
	mma->DesignChange(...)

}

*/

class MMA{
public:

	// Construct using defaults subproblem penalization
	MMA(PetscInt n, PetscInt m, Vec x);
	// User defined subproblem penalization
	MMA(PetscInt n, PetscInt m, Vec x, PetscScalar *a,PetscScalar *c, PetscScalar *d);
	// Initialize with restart from itr
	MMA(PetscInt n, PetscInt m, PetscInt itr, Vec xo1,Vec xo2,Vec U,Vec L);
	// Destructor
	~MMA();

	 // Set and solve a subproblem: return new xval
    PetscErrorCode Update(Vec xval, Vec dfdx, PetscScalar *gx, Vec *dgdx, Vec xmin, Vec xmax);

	// Return necessary data for possible restart
	PetscErrorCode Restart(Vec xo1,Vec xo2,Vec U,Vec L);

	// Set the aggresivity of the moving asymptotes
	PetscErrorCode SetAsymptotes(PetscScalar init, PetscScalar decrease, PetscScalar increase);

	// do/don't add convexity approx to constraints: default=false
    PetscErrorCode ConstraintModification(PetscBool conMod){constraintModification = conMod; return 0;};

	// val=0: default, val=1: increase robustness, i.e
    // control the spacing between L < alp < x < beta < U,
	PetscErrorCode SetRobustAsymptotesType(PetscInt val);

	// Sets outer move limits on all primal design variables
	// This is often requires to prevent the solver from oscilating
	PetscErrorCode SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax, PetscScalar movelim, Vec x, Vec xmin, Vec xmax);

    PetscErrorCode SetOuterMovelimitVec(Vec XMIN, Vec XMAX, PetscScalar movelim, Vec x, Vec xmin, Vec xmax);

	// Return KKT residual norms (norm2 and normInf)
	PetscErrorCode KKTresidual(Vec xval, Vec dfdx, PetscScalar *gx, Vec *dgdx, Vec xmin, Vec xmax,
			PetscScalar *norm2, PetscScalar *normInf);

	// Inf norm on diff between two vectors: SHOULD NOT BE HERE - USE BASIC PETSc!!!!!
	PetscScalar DesignChange(Vec x, Vec xold);

private:

	// Set up the MMA subproblem based on old x's and xval
        PetscErrorCode GenSub(Vec xval, Vec dfdx, PetscScalar *gx, Vec *dgdx, Vec xmin, Vec xmax);

	// Interior point solver for the subproblem
	PetscErrorCode SolveDIP(Vec xval);

	// Compute primal vars based on dual solution
	PetscErrorCode XYZofLAMBDA(Vec x);

	// Dual gradient
	PetscErrorCode DualGrad(Vec x);

	// Dual Hessian
	PetscErrorCode DualHess(Vec x);

	// Dual line search
	PetscErrorCode DualLineSearch();

	// Dual residual
	PetscScalar DualResidual(Vec x, PetscScalar epsi);

	// Problem size and iteration counter
	PetscInt n,m,k;

	// "speed-control" for the asymptotes
	PetscScalar asyminit,asymdec,asyminc;

	// do/don't add convexity constraint approximation in subproblem
        PetscBool constraintModification; // default = FALSE

	// Bool specifying if non lin constraints are included or not
	PetscBool NonLinConstraints;

	// 0: (default) span between alp L x U beta,
	// 1: increase the span for further robustness
	PetscInt RobustAsymptotesType;

	// Local vectors: penalty numbers for subproblem
	PetscScalar *a, *c, *d;

	// Local vectors: elastic variables
	PetscScalar *y;
	PetscScalar z;

	// Local vectors: Lagrange multipliers:
	PetscScalar *lam,*mu,*s;

	// Global: Asymptotes, bounds, objective approx., constraint approx.
	Vec L,U,alpha,beta,p0,q0,*pij,*qij;

	// Local: subproblem constant terms, dual gradient, dual hessian
	PetscScalar *b,*grad,*Hess;

	// Global: Old design variables
        Vec xo1, xo2;

	// Math helpers
	PetscErrorCode Factorize(PetscScalar *K, PetscInt nn);
	PetscErrorCode Solve(PetscScalar *K, PetscScalar *x, PetscInt nn);
	PetscScalar Min(PetscScalar d1, PetscScalar d2);
	PetscScalar Max(PetscScalar d1, PetscScalar d2);
	PetscInt Min(PetscInt d1, PetscInt d2);
	PetscInt Max(PetscInt d1, PetscInt d2);
	PetscScalar Abs(PetscScalar d1);
};


#endif
