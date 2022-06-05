static char help[] = "Project explicit.\n\n";
#include <petscksp.h>
#include <math.h>
#include <petscviewerhdf5.h>
# define FILE "explicit.h5"
#define pi acos(-1)

int main(int argc,char **args)
{
    Vec            u, uold, ua, add_term, save_value;          /* approx solution, RHS, exact solution */
    Mat            A;                /* linear system matrix */
    KSP            ksp;              /* linear solver context */
    PC             pc;               /* preconditioner context */
    PetscReal      CFL=0.4, dx, dt, x=0, t=0;
    PetscErrorCode ierr;
    PetscInt       i,j,n = 100,col[3],its,rstart,rend,nlocal,hstart,hend,hlocal,indic,restart=0;
    PetscScalar    one = 1.0,value[3], v, *array;
    PetscMPIInt    rank;
    PetscViewer    viewer;

   
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-restart",&restart,NULL);CHKERRQ(ierr);
   
    /* set save_value to get n,CFL,t*/
    ierr = VecCreate(PETSC_COMM_WORLD,&save_value);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) save_value, "save_value");
    ierr = VecSetSizes(save_value,3,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(save_value);CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(save_value,&hstart,&hend);CHKERRQ(ierr);
    ierr = VecGetLocalSize(save_value,&hlocal);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"rank = [%d] nlocal = %d hstart = %d hend = %d\n", rank, hlocal,hstart, hend);CHKERRQ(ierr); 
    /* get value from hdf5*/
    if(restart){
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"explicit.h5",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
        ierr = VecLoad(save_value, viewer);CHKERRQ(ierr);
        ierr = VecView(save_value,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        for(i=hstart; i<hend; i+=hlocal){
            col[0] = i;col[1] = i+1;col[2] = i+2;
        }
        ierr = VecGetValues(save_value, 3, col, value);
        // ierr = PetscPrintf(PETSC_COMM_WORLD,"n = %g CFL = %g t = %g\n", value[0], value[1], value[2]);CHKERRQ(ierr);
        n = (int)value[0]; CFL = value[1]; t = value[2]; 
    }
    dx  = 1.0/n;
    dt  = CFL * dx * dx;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dx= %g dt = %g\n", dx, dt);CHKERRQ(ierr);
    // set vector
    // set u_numeracal
    ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "u_numeracal");
    ierr = VecSetSizes(u,PETSC_DECIDE,n+1);CHKERRQ(ierr);
    ierr = VecSetFromOptions(u);CHKERRQ(ierr);
    // set uold and u_analytica, add_term
    ierr = VecDuplicate(u, &uold);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &add_term);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &ua);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ua, "u_analytica");

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
    if(restart){
        VecLoad(u, viewer); // if is reload. Set value here
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr); // close the read IO
    }
    ierr = VecCopy(u,uold);CHKERRQ(ierr);
    ierr = VecView(uold,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
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

    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"explicit.h5",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr); // open write IO
    for (j=0; j<25001; j++)
    {
        t += dt; 
        x =  0;
        // analytical solution
        for (i=0; i<n+1; i++)
        {
            x    = dx*i;
            v    = (PetscReal)(sin(pi*x)/(pi*pi) - sin(pi)*x/(pi*pi));  
            ierr = VecSetValues(ua,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(ua);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(ua);CHKERRQ(ierr);
        //  solution
        ierr = MatMultAdd(A, uold, add_term, u);CHKERRQ(ierr); 
        ierr = VecCopy(u,uold);CHKERRQ(ierr);
        if(!(j%10)){   
            /*add t to save_value */
            for(i=hstart; i<hend; i+=hlocal){
                ierr = VecSetValue(save_value,i,n,INSERT_VALUES);CHKERRQ(ierr);
                ierr = VecSetValue(save_value,i+1,CFL,INSERT_VALUES);CHKERRQ(ierr);
                ierr = VecSetValue(save_value,i+2,t,INSERT_VALUES);CHKERRQ(ierr);
            } 
            ierr = VecAssemblyBegin(save_value);CHKERRQ(ierr);
            ierr = VecAssemblyEnd(save_value);CHKERRQ(ierr); 
            /*print*/
            ierr = PetscPrintf(PETSC_COMM_WORLD,"t = %f\n", t);CHKERRQ(ierr);
            // write numerical solution and n,t,CFL in hdf5
            ierr = PetscPrintf(PETSC_COMM_WORLD,"n = %d CFL = %f t = %f\n", n, CFL, t);CHKERRQ(ierr);
            ierr = VecView(save_value,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = VecView(save_value,viewer);CHKERRQ(ierr);
            ierr = VecView(u,viewer);CHKERRQ(ierr);          
        }
    }ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
   
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = VecDestroy(&uold);CHKERRQ(ierr); 
    ierr = VecDestroy(&ua);CHKERRQ(ierr); 
    ierr = VecDestroy(&add_term);CHKERRQ(ierr); 
    ierr = VecDestroy(&save_value);CHKERRQ(ierr); 
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}

