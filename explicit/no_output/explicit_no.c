static char help[] = "Project explicit.\n\n";
#include <petscksp.h>
#include <math.h>
#include <petscviewerhdf5.h>
#include <sys/time.h>
# define FILE "explicit.h5"
#define pi acos(-1)
 
int main(int argc,char **args)
{
    Vec            u, uold, ua, add_term, err_term;          /* approx solution, RHS, exact solution */
    Mat            A;                /* linear system matrix */
    KSP            ksp;              /* linear solver context */
    PC             pc;               /* preconditioner context */
    PetscReal      CFL=0.5, dx, dt, x=0, t=0, timeuse;
    PetscErrorCode ierr;
    PetscInt       i,j,n = 100,col[3],its,rstart,rend,nlocal;
    PetscScalar    one = 1.0,value[3], v, *array;
    PetscMPIInt    rank;
    PetscViewer    viewer;
    struct timeval start,end;  
    struct timeval {  
    long tv_sec; // 秒数  
    long tv_usec; //微秒数  
    }; 
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    /* get input value*/
    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-CFL",&CFL,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
   
   
    dx  = 1.0/n;
    // dt  = CFL * dx * dx;
    CFL = dt/(dx*dx);
    its = (int) 1.0/dt + 2;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dx= %g dt = %g its %d CFL %g\n", dx, dt, its, CFL);CHKERRQ(ierr);
    // set vector
    // set u_numeracal
    ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "u_numeracal");
    ierr = VecSetSizes(u,PETSC_DECIDE,n+1);CHKERRQ(ierr);
    ierr = VecSetFromOptions(u);CHKERRQ(ierr);
    // set uold and u_analytica, add_term
    ierr = VecDuplicate(u, &uold);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &add_term);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &err_term);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &ua);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ua, "u_analytica");

    ierr = VecGetOwnershipRange(u,&rstart,&rend);CHKERRQ(ierr);
    ierr = VecGetLocalSize(u,&nlocal);CHKERRQ(ierr);
    gettimeofday(&start, NULL ); // Record the start time of matrix packaging.
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
    gettimeofday(&end, NULL );  // Record the end time of matrix packaging.
    timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;  
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Mat time cost = %f s\n", timeuse/1000000.0);CHKERRQ(ierr);        

    // set initial value
    /* set origin u and uold = u */
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
     /* set add_term */
    ierr = VecSetValue(add_term,0,0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(add_term,n,0,INSERT_VALUES);CHKERRQ(ierr);
    for (i=1; i<n; i++) 
    {
        x    = i*dx;
        v    = (PetscReal)(dt*sin(pi*x)); 
        ierr = VecSetValues(add_term,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(add_term);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(add_term);CHKERRQ(ierr);
    // analytical solution
    // for (i=0; i<n+1; i++)
    // {
    //     x    = dx*i;
    //     v    = (PetscReal)(sin(pi*x)/(pi*pi) - sin(pi)*x/(pi*pi));  
    //     ierr = VecSetValues(ua,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    // }
    // ierr = VecAssemblyBegin(ua);CHKERRQ(ierr);
    // ierr = VecAssemblyEnd(ua);CHKERRQ(ierr);

    gettimeofday(&start, NULL ); // Record the start time of matrix packaging.
    for (j=0; j<its; j++)
    {
        t += dt; 
        x =  0;
        //  solution
        ierr = MatMultAdd(A, uold, add_term, u);CHKERRQ(ierr); 
        ierr = VecCopy(u,uold);CHKERRQ(ierr);
        ierr = VecCopy(u,err_term);CHKERRQ(ierr);
        // if(!(j%10)){   
        //     /*print*/
        //     ierr = PetscPrintf(PETSC_COMM_WORLD,"t = %f\n", t);CHKERRQ(ierr);
        //     // write numerical solution and n,t,CFL in hdf5
        //     ierr = PetscPrintf(PETSC_COMM_WORLD,"n = %d CFL = %f t = %f\n", n, CFL, t);CHKERRQ(ierr);      
        // }
    }
    gettimeofday(&end, NULL );  // Record the end time of matrix packaging.
    timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;  
    ierr = PetscPrintf(PETSC_COMM_WORLD,"iteration time cost = %f s\n", timeuse/1000000.0);CHKERRQ(ierr);        

    ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = VecDestroy(&uold);CHKERRQ(ierr); 
    ierr = VecDestroy(&ua);CHKERRQ(ierr); 
    ierr = VecDestroy(&add_term);CHKERRQ(ierr); 
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}

