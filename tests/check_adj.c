#include <stdlib.h>
#include <complex.h>
#include <pnfft.h>

static void perform_pnfft_adj_guru(
    const ptrdiff_t *N, const ptrdiff_t *n, ptrdiff_t local_M,
    int m, const double *x_max,
    unsigned pnfft_flags, unsigned compute_flags,
    const int *np, MPI_Comm comm, const char *name,
    pnfft_complex **f_hat, double *f_hat_sum, ptrdiff_t *local_N_total);

static void compare_f_hat(
    const pnfft_complex *f_hat_pnfft, const pnfft_complex *f_hat_nfft, ptrdiff_t local_N_total,
    double f_hat_sum, const char *name, MPI_Comm comm_cart_3d);


int main(int argc, char **argv){
  int np[3], m,  compare_direct=0, debug;
  unsigned pnfft_flags;
  ptrdiff_t N[3], n[3], local_M, local_N_total;
  double f_hat_sum, x_max[3];
  unsigned compute_flags;
  pnfft_complex *f_hat1=NULL, *f_hat2=NULL;
  
  MPI_Init(&argc, &argv);
  pnfft_init();
  
  /* set values by commandline */
  pnfft_check_init_parameters(argc, argv, N, n, &local_M, &m, &pnfft_flags, &compute_flags,
      x_max, np, &compare_direct, &debug);

  /* calculate parallel NFFT */
  perform_pnfft_adj_guru(N, n, local_M, m,   x_max, pnfft_flags, compute_flags,
      np, MPI_COMM_WORLD, "PNFFT adj",
      &f_hat1, &f_hat_sum, &local_N_total);

  /* calculate parallel NDFT or NFFT with higher accuracy */
  if(compare_direct) compute_flags |= PNFFT_COMPUTE_DIRECT;
  else               m += 2;

  perform_pnfft_adj_guru(N, n, local_M, m,   x_max, pnfft_flags, compute_flags,
      np, MPI_COMM_WORLD, "reference method",
      &f_hat2, &f_hat_sum, &local_N_total);

  /* calculate error of PNFFT */
  compare_f_hat(f_hat1, f_hat2, local_N_total, f_hat_sum, "* Results in f_hat", MPI_COMM_WORLD);

  /* free mem and finalize */
  pnfft_free(f_hat1); pnfft_free(f_hat2);
  pnfft_cleanup();
  MPI_Finalize();
  return 0;
}


static void perform_pnfft_adj_guru(
    const ptrdiff_t *N, const ptrdiff_t *n, ptrdiff_t local_M,
    int m, const double *x_max, 
    unsigned pnfft_flags, unsigned compute_flags,
    const int *np, MPI_Comm comm, const char *name,
    pnfft_complex **f_hat, double *f_hat_sum, ptrdiff_t *local_N_total
    )
{
  int myrank;
  ptrdiff_t local_N[3], local_N_start[3];
  double lower_border[3], upper_border[3];
  double local_sum = 0, time, time_max;
  MPI_Comm comm_cart_3d;
  pnfft_complex *f, *grad_f;
  double *x;
  pnfft_plan pnfft;
  pnfft_nodes nodes;

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

  *local_N_total = 1;
  for(int t=0; t<3; t++)
    *local_N_total *= local_N[t];

  /* plan parallel NFFT */
  pnfft = pnfft_init_guru(3, N, n, x_max, m,
      PNFFT_MALLOC_F_HAT | pnfft_flags, PFFT_ESTIMATE,
      comm_cart_3d);

  /* initialize nodes */
  unsigned malloc_flags = PNFFT_MALLOC_X;
  if(compute_flags & PNFFT_COMPUTE_F)       malloc_flags |= PNFFT_MALLOC_F;
  if(compute_flags & PNFFT_COMPUTE_GRAD_F)  malloc_flags |= PNFFT_MALLOC_GRAD_F;

  nodes = pnfft_init_nodes(local_M, malloc_flags);

  /* get data pointers */
  *f_hat = pnfft_get_f_hat(pnfft);
  f      = pnfft_get_f(nodes);
  grad_f = pnfft_get_grad_f(nodes);
  x      = pnfft_get_x(nodes);

  /* initialize Fourier coefficients */
  srand(myrank);

  if(compute_flags & PNFFT_COMPUTE_F)
    pnfft_init_f(local_M,
        f);

  if(compute_flags & PNFFT_COMPUTE_GRAD_F)
    pnfft_init_f(3*local_M,
        grad_f);

  /* initialize nonequispaced nodes */
  pnfft_init_x_3d_adv(lower_border, upper_border, x_max, local_M,
      x);

  /* execute parallel NFFT */
  time = -MPI_Wtime();
  pnfft_adj(pnfft, nodes, compute_flags);
  time += MPI_Wtime();
  
  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s needs %6.2e s\n", name, time_max);
 
  /* calculate norm of Fourier coefficients for calculation of relative error */ 
  for(ptrdiff_t k=0; k<local_N[0]*local_N[1]*local_N[2]; k++)
    local_sum += cabs((*f_hat)[k]);
  MPI_Allreduce(&local_sum, f_hat_sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart_3d);

  /* free mem and finalize, do not free nfft.f_hat */
  pnfft_finalize(pnfft, PNFFT_FREE_NONE);
  pnfft_free_nodes(nodes, malloc_flags);
  MPI_Comm_free(&comm_cart_3d);
}


static void compare_f_hat(
    const pnfft_complex *f_hat_pnfft, const pnfft_complex *f_hat_nfft, ptrdiff_t local_N_total,
    double f_hat_sum, const char *name, MPI_Comm comm
    )
{
  double error = 0, error_max;

  for(ptrdiff_t j=0; j<local_N_total; j++)
    if( cabs(f_hat_pnfft[j]-f_hat_nfft[j]) > error)
      error = cabs(f_hat_pnfft[j]-f_hat_nfft[j]);

  MPI_Reduce(&error, &error_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s - absolute error = %6.2e,  relative error =  %6.2e\n", name, error_max, error_max/f_hat_sum);
}

