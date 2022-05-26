static char help[] = "Project.\n\n";
#include <petscksp.h>
#include <math.h>
#define pi acos(-1)

int main(int argc,char **args)
{
    Vec            u, uold;          /* approx solution, RHS, exact solution */
    Mat            A;                /* linear system matrix */
    KSP            ksp;              /* linear solver context */
    PC             pc;               /* preconditioner context */
    PetscReal      CFL, dx, dt, x, t;
    PetscErrorCode ierr;
    PetscInt       i,j,n = 100,col[3],its,rstart,rend,nlocal;
    PetscScalar    one = 1.0,value[3], v, *array;
    PetscMPIInt    rank;

   
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
 
    x   = 0;
    t   = 0;
    CFL = 0.4;
    dx  = 1.0/n;
    dt  = CFL * dx * dx;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dx= %g dt = %g\n", dx, dt);CHKERRQ(ierr);

    // set vector
    ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
    ierr = VecSetSizes(u,PETSC_DECIDE,n+1);CHKERRQ(ierr);
    ierr = VecSetFromOptions(u);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &uold);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(u,&rstart,&rend);CHKERRQ(ierr);
    ierr = VecGetLocalSize(u,&nlocal);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_SELF,"rank = [%d] nlocal = %d rstart = %d rend = %d\n", rank, nlocal,rstart, rend);CHKERRQ(ierr); 
    // set matrix
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,nlocal,nlocal,n+1,n+1);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    if (!rstart) 
        rstart = 1;
    if (rend == (n+1)) 
        rend = n;
    /* Set entries corresponding to the mesh interior */
    value[0] = CFL; value[1] = 1 - 2*CFL; value[2] = CFL;
    for (i=rstart; i<rend; i++) 
    {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }  
    /* Assemble the matrix */
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    // set initial value
    ierr = VecSetValue(u,0,0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(u,n,0,INSERT_VALUES);CHKERRQ(ierr);
    for (i=1; i<n; i++) 
    {
    x = i*dx;
    v    = (PetscReal)(exp(x));
    ierr = VecSetValues(u,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
    ierr = VecCopy(u,uold);CHKERRQ(ierr);
    ierr = VecView(uold,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    for (j=0; j<25001; j++)
    {
        t += dt; 
        x = 0;
        ierr = MatMult(A, uold, u);
        for (i=1; i<n ; i++)
        {
            x   += dx;
            v    = (PetscReal)(dt*sin(pi*x));  
            // ierr = PetscPrintf(PETSC_COMM_SELF,"v = %g\n", v);CHKERRQ(ierr);
            ierr = VecSetValues(u,1,&i,&v,ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
        ierr = VecCopy(u,uold);CHKERRQ(ierr);
        if(!(j%200)){
            ierr = PetscPrintf(PETSC_COMM_WORLD,"t = %g\n", t);CHKERRQ(ierr);
            ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        }
    }
    ierr = PetscFinalize();
    return ierr;
}

