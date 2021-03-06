static char help[] = "Project implicit.\n\n";
#include <petscksp.h>
#include <math.h>
#include <petscviewerhdf5.h>
#define pi acos(-1)
# define FILE "implicit.h5"

int main(int argc,char **args)
{
    Vec            u, uold, ua, add_term, err_term, save_value;          /* approx solution, RHS, exact solution */
    Mat            A;                /* linear system matrix */
    KSP            ksp;              /* linear solver context */
    PC             pc;               /* preconditioner context */
    PetscReal      CFL=0.4, dx, dt, x=0, t=0;
    PetscErrorCode ierr;
    PetscInt       i,j,n = 100,col[3],its,rstart,rend,nlocal, hstart,hend,hlocal,restart=0;
    PetscScalar    one = 1.0,value[3], v, *array;
    PetscMPIInt    rank;
    PetscViewer    viewer;

   
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    /* get input value*/
    ierr = PetscOptionsGetInt(NULL,NULL,"-restart",&restart,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
    /* set vector save_value to store n,CFL,t*/
    ierr = VecCreate(PETSC_COMM_WORLD,&save_value);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) save_value, "save_value");
    ierr = VecSetSizes(save_value,3,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(save_value);CHKERRQ(ierr);
     /* git the start and end of save_value in each cpu */
    ierr = VecGetOwnershipRange(save_value,&hstart,&hend);CHKERRQ(ierr);
    ierr = VecGetLocalSize(save_value,&hlocal);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"rank = [%d] nlocal = %d hstart = %d hend = %d\n", rank, hlocal,hstart, hend);CHKERRQ(ierr); 
    /* get value from hdf5*/
    if(restart){
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"implicit.h5",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
        ierr = VecLoad(save_value, viewer);CHKERRQ(ierr);
        ierr = VecView(save_value,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        for(i=hstart; i<hend; i+=hlocal){
            col[0] = i;col[1] = i+1;col[2] = i+2;
        }
        ierr = VecGetValues(save_value, 3, col, value);
        n = (int)value[0]; CFL = value[1]; t = value[2]; 
    }
    
    dx  = 1.0/n;
    dt  = CFL * dx * dx;
    its = (int) 1.0/dt + 2;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dx= %g dt = %g its %d\n", dx, dt, its);CHKERRQ(ierr);

    // set vector
    ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "u_numeracal");
    ierr = VecSetSizes(u,PETSC_DECIDE,n+1);CHKERRQ(ierr);
    ierr = VecSetFromOptions(u);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &uold);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &ua);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &add_term);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &err_term);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(u,&rstart,&rend);CHKERRQ(ierr);
    ierr = VecGetLocalSize(u,&nlocal);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_SELF,"rank = [%d] nlocal = %d rstart = %d rend = %d\n", rank, nlocal,rstart, rend);CHKERRQ(ierr); 
    // set matrix
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,nlocal,nlocal,n+1,n+1);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    if (!rstart)
    { 
        col[0] = 0; value[0] = 1;
        ierr   = MatSetValues(A,1,&rstart,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
        rstart = 1;
    }        
    if (rend == (n+1))
    { 
        rend = n;
        col[0] = n; value[0] = 1;
        ierr   = MatSetValues(A,1,&rend,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* Set entries corresponding to the mesh interior */
    value[0] = -CFL; value[1] = 1 + 2*CFL; value[2] = -CFL;
    for (i=rstart; i<rend; i++) 
    {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }  
    /* Assemble the matrix */
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    // set initial vector value
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
    /*if reload, Load vector u in the file*/
    if(restart){
        VecLoad(u, viewer); // if is reload. Set value here
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr); // close the read IO
    }
    for (i=0; i<n+1; i++) 
    {
        x = i*dx;
        v    = (PetscReal)(dt*sin(pi*x));
        ierr = VecSetValues(add_term,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(add_term);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(add_term);CHKERRQ(ierr);

    ierr = VecAXPY(u,1.0,add_term);CHKERRQ(ierr);
    ierr = VecCopy(u,uold);CHKERRQ(ierr);
    ierr = VecView(uold,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* ksp solve set */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc, PCJACOBI);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"implicit.h5",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr); // open write IO
    // solve
    for (j=0; j<its; j++)
    {
        t += dt; 
        x  = 0;
        // analytical solution
        for (i=0; i<n+1; i++)
        {
            x    = dx*i;
            v    = (PetscReal)(sin(pi*x)/(pi*pi) - sin(pi)*x/(pi*pi));  
            // ierr = PetscPrintf(PETSC_COMM_SELF,"v = %g\n", v);CHKERRQ(ierr);
            ierr = VecSetValues(ua,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(ua);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(ua);CHKERRQ(ierr);
        //  solution
        x = 0;
        ierr = KSPSolve(ksp, uold, u);
        ierr = VecCopy(u,uold);CHKERRQ(ierr);
        ierr = VecAXPY(uold,1.0,add_term);CHKERRQ(ierr);
        ierr = VecCopy(u,err_term);CHKERRQ(ierr);
        if(!(j%10)){
             /*add t to save_value */
            for(i=hstart; i<hend; i+=hlocal){
                ierr = VecSetValue(save_value,i,n,INSERT_VALUES);CHKERRQ(ierr);
                ierr = VecSetValue(save_value,i+1,CFL,INSERT_VALUES);CHKERRQ(ierr);
                ierr = VecSetValue(save_value,i+2,t,INSERT_VALUES);CHKERRQ(ierr);
            } 
            ierr = PetscPrintf(PETSC_COMM_WORLD,"t = %g\n", t);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"numercial solution");CHKERRQ(ierr);
            ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"analytica solutionl");CHKERRQ(ierr);
            ierr = VecView(ua,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = VecAXPY(err_term,-1.0,ua);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"error term");CHKERRQ(ierr);
            ierr = VecView(err_term,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = VecView(save_value,viewer);CHKERRQ(ierr);
            ierr = VecView(u,viewer);CHKERRQ(ierr);   
        }
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = VecDestroy(&uold);CHKERRQ(ierr); 
    ierr = VecDestroy(&ua);CHKERRQ(ierr); 
    ierr = VecDestroy(&add_term);CHKERRQ(ierr); 
    ierr = VecDestroy(&err_term);CHKERRQ(ierr); 
    ierr = VecDestroy(&save_value);CHKERRQ(ierr); 
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}

