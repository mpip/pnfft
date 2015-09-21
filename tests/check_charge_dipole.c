#include <stdlib.h>
#include <complex.h>
#include <pnfft.h>

static void pnfft_perform_guru(
    const ptrdiff_t *N, const ptrdiff_t *n, ptrdiff_t local_M,
    int m, const double *x_max, 
    unsigned pnfft_flags, unsigned compute_flags,
    const int *np, MPI_Comm comm, const char *name,
    pnfft_complex **f, pnfft_complex **grad_f, pnfft_complex **hessian_f,
    double *f_hat_sum);

static void compare_f(
    const pnfft_complex *f1, const pnfft_complex *f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm);
static void compare_grad_f(
    const pnfft_complex *grad_f1, const pnfft_complex *grad_f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm);
static void compare_hessian_f(
    const pnfft_complex *hessian_f1, const pnfft_complex *hessian_f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm);


int main(int argc, char **argv){
  ptrdiff_t local_M, local_Mc, local_Md;
  pnfft_complex *charges1=NULL, *charges2=NULL;
  pnfft_complex *dipoles1=NULL, *dipoles2=NULL;


  local_Mc = 100; // local num. of charges
  local_Md = 50;  // local num. of dipoles
  local_M = local_Mc + local_Md;


  int np[3], m, compare_direct=0, debug;
  unsigned pnfft_flags;
  ptrdiff_t N[3], n[3], local_M;
  double f_hat_sum, x_max[3];
  unsigned compute_flags;
  pnfft_complex *f1=NULL, *f2=NULL;
  pnfft_complex *grad_f1=NULL, *grad_f2=NULL;
  pnfft_complex *hessian_f1=NULL, *hessian_f2=NULL;
  
  MPI_Init(&argc, &argv);
  pnfft_init();
  
  /* set values by commandline */
  pnfft_check_init_parameters(argc, argv, N, n, &local_M, &m, &pnfft_flags, &compute_flags,
      x_max, np, &compare_direct, &debug);

  /* calculate parallel NFFT */
  pnfft_perform_guru(N, n, local_M, m,   x_max, pnfft_flags, compute_flags,
      np, MPI_COMM_WORLD, "PNFFT trafo",
      &f1, &grad_f1, &hessian_f1, &f_hat_sum);

  /* calculate parallel NDFT or NFFT with higher accuracy */
  if(compare_direct) compute_flags |= PNFFT_COMPUTE_DIRECT;
  else               m += 2;

  pnfft_perform_guru(N, n, local_M, m, x_max, pnfft_flags, compute_flags,
      np, MPI_COMM_WORLD, "reference method",
      &f2, &grad_f2, &hessian_f2, &f_hat_sum);

  /* calculate error of PNFFT */
  compare_f(f1, f2, local_M, f_hat_sum, "* Results in f", MPI_COMM_WORLD);
  compare_grad_f(grad_f1, grad_f2, local_M, f_hat_sum, "* Results in grad_f", MPI_COMM_WORLD);
  compare_hessian_f(hessian_f1, hessian_f2, local_M, f_hat_sum, "* Results in hessian_f", MPI_COMM_WORLD);



  compare_energy(energy1, energy2, local_M, "* Results in energy", MPI_COMM_WORLD);
  compare_force( force1,  force2,  local_M, "* Results in force ", MPI_COMM_WORLD);

  /* free mem and finalize */
  if(f1)         pnfft_free(f1);
  if(f2)         pnfft_free(f2);
  if(grad_f1)    pnfft_free(grad_f1);
  if(grad_f2)    pnfft_free(grad_f2);
  if(hessian_f1) pnfft_free(hessian_f1);
  if(hessian_f2) pnfft_free(hessian_f2);
  pnfft_cleanup();
  MPI_Finalize();
  return 0;
}


static void pnfft_perform_guru(
    const ptrdiff_t *N, const ptrdiff_t *n,
    ptrdiff_t local_Mc, ptrdiff_t local_Md,
    int m, const double *x_max, 
    unsigned pnfft_flags, unsigned compute_flags,
    const int *np, MPI_Comm comm, const char *name,
    pnfft_complex **energy, pnfft_complex **force,
    double *f_hat_sum
    )
{
  int myrank;
  ptrdiff_t local_N[3], local_N_start[3];
  double lower_border[3], upper_border[3];
  double local_sum = 0, time, time_max;
  MPI_Comm comm_cart_3d;
  pnfft_complex *f_hat;
  double *charges_x, *dipoles_x;
  pnfft_complex *charges_val, *dipoles_val;
  pnfft_plan pnfft;
  pnfft_nodes charges, dipoles;
  pnfft_complex *buffer;

  /* create three-dimensional process grid of size np[0] x np[1] x np[2], if possible */
  if( pnfft_create_procmesh(3, comm, np, &comm_cart_3d) ){
    pfft_fprintf(comm, stderr, "Error: Procmesh of size %d x %d x %d does not fit to number of allocated processes.\n", np[0], np[1], np[2]);
    pfft_fprintf(comm, stderr, "       Please allocate %d processes (mpiexec -np %d ...) or change the procmesh (with -pnfft_np * * *).\n", np[0]*np[1]*np[2], np[0]*np[1]*np[2]);
    MPI_Finalize();
    return;
  }

  MPI_Comm_rank(comm_cart_3d, &myrank);

  /* get parameters of data distribution */
  pnfft_local_size_guru(3, N, n, x_max, m, comm_cart_3d, pnfft_flags & PNFFT_TRANSPOSED_F_HAT,
      local_N, local_N_start, lower_border, upper_border);

  /* plan parallel NFFT */
  pnfft = pnfft_init_guru(3, N, n, x_max, m,
      PNFFT_MALLOC_F_HAT | pnfft_flags, PFFT_ESTIMATE,
      comm_cart_3d);

  /* initialize nodes */
  charges = pnfft_init_nodes(local_Mc, PNFFT_MALLOC_X | PNFFT_MALLOC_F | PNFFT_MALLOC_GRAD_F);
  dipoles = pnfft_init_nodes(local_Md, PNFFT_MALLOC_X | PNFFT_MALLOC_GRAD_F | PNFFT_MALLOC_HESSIAN_F);

  /* get data pointers */
  f_hat       = pnfft_get_f_hat(pnfft);
  charges_val = pnfft_get_f(charges);
  charges_x   = pnfft_get_x(charges);
  dipoles_val = pnfft_get_grad_f(dipoles);
  dipoles_x   = pnfft_get_x(dipoles);
//   *grad_f    = pnfft_get_grad_f(charges);
//   *hessian_f = pnfft_get_hessian_f(nodes);

  /* initialize charges and dipoles */
  srand(myrank);

  pnfft_init_f(local_Mc,
      charges_val);

  pnfft_init_f(3*local_Md,
      dipoles_val);

  /* initialize nonequispaced nodes */
  srand(myrank+1);

  pnfft_init_x_3d_adv(lower_border, upper_border, x_max, local_Mc,
      charges_x);

  pnfft_init_x_3d_adv(lower_border, upper_border, x_max, local_Md,
      dipoles_x);

  /* execute parallel adj. NFFT */
  time = -MPI_Wtime();
  pnfft_adj(pnfft, charges, PNFFT_COMPUTE_F);
  time += MPI_Wtime();
  
  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s needs %6.2e s\n", name, time_max);

  /* TODO: use PNFFT_COMPUTE_ACCUMULATED to avoid extra memory for buffer */
  buffer = pnfft_malloc_complex(local_N[0]*local_N[1]*local_N[2]);
  for(ptrdiff_t k=0; k<local_N[0]*local_N[1]*local_N[2]; k++)
    buffer[k] = f_hat[k];

  /* execute parallel adj. NFFT */
  time = -MPI_Wtime();
  pnfft_adj(pnfft, dipoles, PNFFT_COMPUTE_GRAD_F);
  time += MPI_Wtime();
  
  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s needs %6.2e s\n", name, time_max);

  for(ptrdiff_t k=0; k<local_N[0]*local_N[1]*local_N[2]; k++)
    f_hat[k] += buffer[k];
 
  /* calculate norm of Fourier coefficients for calculation of relative error */ 
  for(ptrdiff_t k=0; k<local_N[0]*local_N[1]*local_N[2]; k++)
    local_sum += cabs(f_hat[k]);
  MPI_Allreduce(&local_sum, f_hat_sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart_3d);


  /* TODO: add NFFT part here */



  /* TODO: compute energy and force */


  /* free mem and finalize, do not free nfft.f */
  pnfft_finalize(pnfft, PNFFT_FREE_F_HAT);
  pnfft_free_nodes(nodes, PNFFT_FREE_X);
  MPI_Comm_free(&comm_cart_3d);
  pnfft_free(buffer);
}

static void compare_f(
    const pnfft_complex *f1, const pnfft_complex *f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm
    )
{
  if(f1==NULL) return;
  if(f2==NULL) return;

  double error = 0, error_max;

  for(ptrdiff_t j=0; j<local_M; j++)
    if( cabs(f1[j]-f2[j]) > error)
      error = cabs(f1[j]-f2[j]);

  MPI_Reduce(&error, &error_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s - absolute error = %6.2e,  relative error =  %6.2e\n", name, error_max, error_max/f_hat_sum);
}

static void compare_grad_f(
    const pnfft_complex *grad_f1, const pnfft_complex *grad_f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm
    )
{
  if(grad_f1==NULL) return;
  if(grad_f2==NULL) return;

  double error, error_max;

  for(int t=0; t<3; t++){
    error = 0;
    for(ptrdiff_t j=0; j<local_M; j++)
      if( cabs(grad_f1[3*j+t]-grad_f2[3*j+t]) > error)
        error = cabs(grad_f1[3*j+t]-grad_f2[3*j+t]);
    MPI_Reduce(&error, &error_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    pfft_printf(comm, "%s, %d. component - absolute error = %6.2e,  relative error =  %6.2e\n", name, t, error_max, error_max/f_hat_sum);
  }
}

static void compare_hessian_f(
    const pnfft_complex *hessian_f1, const pnfft_complex *hessian_f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm
    )
{
  if(hessian_f1==NULL) return;
  if(hessian_f2==NULL) return;

  double error, error_max;

  for(int t=0; t<6; t++){
    error = 0;
    for(ptrdiff_t j=0; j<local_M; j++)
      if( cabs(hessian_f1[6*j+t]-hessian_f2[6*j+t]) > error)
        error = cabs(hessian_f1[6*j+t]-hessian_f2[6*j+t]);
    MPI_Reduce(&error, &error_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    pfft_printf(comm, "%s, %d. component - absolute error = %6.2e,  relative error =  %6.2e\n", name, t, error_max, error_max/f_hat_sum);
  }
}

