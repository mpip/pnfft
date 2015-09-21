#include <stdlib.h>
#include <complex.h>
#include <pnfft.h>

static void pnfft_perform_guru(
    const ptrdiff_t *N, const ptrdiff_t *n,
    ptrdiff_t local_Mc, ptrdiff_t local_Md,
    int m, const double *x_max, 
    unsigned pnfft_flags, unsigned compute_flags,
    const int *np, MPI_Comm comm, const char *name,
    pnfft_complex **energy, pnfft_complex **force);

static void compare_energy(
    const pnfft_complex *e1, const pnfft_complex *e2, ptrdiff_t local_M,
    const char *name, MPI_Comm comm);
static void compare_force(
    const pnfft_complex *f1, const pnfft_complex *f2, ptrdiff_t local_M,
    const char *name, MPI_Comm comm);


int main(int argc, char **argv){
  ptrdiff_t local_M, local_Mc, local_Md;
  int np[3], m, compare_direct=0, debug;
  unsigned pnfft_flags;
  ptrdiff_t N[3], n[3];
  double x_max[3];
  unsigned compute_flags;
  pnfft_complex *energy1=NULL, *energy2=NULL;
  pnfft_complex *force1=NULL,  *force2=NULL;
  
  MPI_Init(&argc, &argv);
  pnfft_init();
  
  /* set values by commandline */
  pnfft_check_init_parameters(argc, argv, N, n, &local_M, &m, &pnfft_flags, &compute_flags,
      x_max, np, &compare_direct, &debug);

  local_Mc = local_M/2;
  local_Md = local_M - local_Mc;

  compute_flags &= PNFFT_TRANSPOSED_F_HAT | PNFFT_COMPUTE_DIRECT;

  /* calculate parallel NFFT */
  pnfft_perform_guru(N, n, local_Mc, local_Md, m, x_max, pnfft_flags, compute_flags,
      np, MPI_COMM_WORLD, "PNFFT",
      &energy1, &force1);

  /* calculate parallel NDFT or NFFT with higher accuracy */
  if(compare_direct) compute_flags |= PNFFT_COMPUTE_DIRECT;
  else               m += 2;

  pnfft_perform_guru(N, n, local_Mc, local_Md, m, x_max, pnfft_flags, compute_flags,
      np, MPI_COMM_WORLD, "reference method",
      &energy2, &force2);

  /* calculate error of PNFFT */
  compare_energy(energy1, energy2, local_M, "* Results in energy", MPI_COMM_WORLD);
  compare_force( force1,  force2,  local_M, "* Results in force", MPI_COMM_WORLD);

  /* free mem and finalize */
  if(energy1) pnfft_free(energy1);
  if(energy2) pnfft_free(energy2);
  if(force1)  pnfft_free(force1);
  if(force2)  pnfft_free(force2);
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
    pnfft_complex **energy, pnfft_complex **force
    )
{
  int myrank;
  ptrdiff_t local_N[3], local_N_start[3];
  double lower_border[3], upper_border[3];
  double time, time_max;
  MPI_Comm comm_cart_3d;
  pnfft_plan pnfft;
  pnfft_nodes charges, dipoles;
  pnfft_complex *buffer;

  *energy = pnfft_alloc_complex(local_Mc + local_Md);
  *force  = pnfft_alloc_complex(3*local_Mc + 3*local_Md);

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

  pnfft_complex *charges_val = local_Mc ? pnfft_alloc_complex(local_Mc)   : NULL;
  pnfft_complex *dipoles_val = local_Mc ? pnfft_alloc_complex(3*local_Md) : NULL;

  /* get data pointers */
  pnfft_complex *f_hat             = pnfft_get_f_hat(pnfft);
  pnfft_complex *charges_f         = pnfft_get_f(charges);
  pnfft_complex *charges_grad_f    = pnfft_get_grad_f(charges);
  double        *charges_x         = pnfft_get_x(charges);
  pnfft_complex *dipoles_grad_f    = pnfft_get_grad_f(dipoles);
  pnfft_complex *dipoles_hessian_f = pnfft_get_hessian_f(dipoles);
  double        *dipoles_x         = pnfft_get_x(dipoles);

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

  for(ptrdiff_t j=0; j<local_Mc; ++j)
    charges_f[j] = charges_val[j];

  for(ptrdiff_t j=0; j<3*local_Md; ++j)
    dipoles_grad_f[j] = dipoles_val[j];

  /* execute parallel adj. NFFT */
  time = -MPI_Wtime();
  pnfft_adj(pnfft, charges, compute_flags | PNFFT_COMPUTE_F);
  time += MPI_Wtime();
  
  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s: adjoint for charges needs %6.2e s\n", name, time_max);

  /* TODO: use PNFFT_COMPUTE_ACCUMULATED to avoid extra memory for buffer */
  buffer = pnfft_alloc_complex(local_N[0]*local_N[1]*local_N[2]);
  for(ptrdiff_t k=0; k<local_N[0]*local_N[1]*local_N[2]; k++)
    buffer[k] = f_hat[k];

  /* execute parallel adj. NFFT */
  time = -MPI_Wtime();
  pnfft_adj(pnfft, dipoles, compute_flags | PNFFT_COMPUTE_GRAD_F);
  time += MPI_Wtime();
  
  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s: adjoint for dipoles needs %6.2e s\n", name, time_max);

  for(ptrdiff_t k=0; k<local_N[0]*local_N[1]*local_N[2]; ++k)
    f_hat[k] += buffer[k];
  pnfft_free(buffer);
 
  /* execute parallel NFFT */
  time = -MPI_Wtime();
  pnfft_trafo(pnfft, charges, compute_flags | PNFFT_COMPUTE_F | PNFFT_COMPUTE_GRAD_F);
  time += MPI_Wtime();

  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s: trafo for charges needs %6.2e s\n", name, time_max);

  time = -MPI_Wtime();
  {
    pnfft_complex *in=charges_val, *out=*energy, *sca=charges_f;
    for(ptrdiff_t j=0; j<local_Mc; ++j)
      out[j] = sca[j] * in[j]; 
  }

  {
    pnfft_complex *in=charges_val, *out=*force, *vec=charges_grad_f;
    for(ptrdiff_t j=0; j<local_Mc; ++j)
      for(ptrdiff_t t=0; t<3; t++)
        out[3*j+t] = vec[3*j+t] * in[j]; 
  }
  time += MPI_Wtime();
  
  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s: Computation of energy and force for charges needs %6.2e s\n", name, time_max);

  time = -MPI_Wtime();
  pnfft_trafo(pnfft, dipoles, compute_flags | PNFFT_COMPUTE_GRAD_F | PNFFT_COMPUTE_HESSIAN_F);
  time += MPI_Wtime();

  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s: trafo for dipoles needs %6.2e s\n", name, time_max);

  time = -MPI_Wtime();
  {
    pnfft_complex *in=dipoles_val, *out=*energy + local_Mc, *vec=dipoles_grad_f;
    for(ptrdiff_t j=0; j<local_Md; ++j){
      out[0] = vec[0] * in[0] + vec[1] * in[1] + vec[2] * in[2];
      ++out; in += 3; vec += 3;
    }
  }

  {
    pnfft_complex *in=dipoles_val, *out=*force + local_Mc, *mat=dipoles_hessian_f;
    for(ptrdiff_t j=0; j<local_Md; ++j){
      /* compute symmetric matrix times vector product */
      out[0] = mat[0] * in[0] + mat[1] * in[1] + mat[2] * in[2];
      out[1] = mat[1] * in[0] + mat[3] * in[1] + mat[4] * in[2];
      out[2] = mat[2] * in[0] + mat[4] * in[1] + mat[5] * in[2];

      out += 3; in += 3; mat += 6;
    }
  }
  time += MPI_Wtime();

  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s: Computation of energy and force for dipoles needs %6.2e s\n", name, time_max);

  /* free mem and finalize, do not free nfft.f */
  pnfft_finalize(pnfft, PNFFT_FREE_F_HAT);
  pnfft_free_nodes(charges, PNFFT_FREE_ALL);
  pnfft_free_nodes(dipoles, PNFFT_FREE_ALL);
  MPI_Comm_free(&comm_cart_3d);
  pnfft_free(charges_val);
  pnfft_free(dipoles_val);
}

static void compare_energy(
    const pnfft_complex *e1, const pnfft_complex *e2, ptrdiff_t local_M,
    const char *name, MPI_Comm comm
    )
{
  if(e1==NULL) return;
  if(e2==NULL) return;

  double error = 0, error_max, tmp;

  double sum=0;
  for(ptrdiff_t j=0; j<local_M; j++){
    tmp = cabs(e2[j]);
    sum += tmp*tmp;

    tmp = cabs(e1[j]-e2[j]);
    if( tmp*tmp > error)
      error = tmp*tmp;
  }

  MPI_Reduce(&error, &error_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s - absolute rms error = %6.2e,  relative rms error =  %6.2e\n", name, sqrt(error_max), sqrt(error_max/sum));
}

static void compare_force(
    const pnfft_complex *f1, const pnfft_complex *f2, ptrdiff_t local_M,
    const char *name, MPI_Comm comm
    )
{
  if(f1==NULL) return;
  if(f2==NULL) return;

  double error, error_max, tmp, sum;

  for(int t=0; t<3; t++){
    sum = 0;
    error = 0;
    for(ptrdiff_t j=0; j<local_M; j++){
      tmp = cabs(f2[3*j+t]);
      sum += tmp*tmp;

      tmp = cabs(f1[3*j+t]-f2[3*j+t]); 
      if( tmp*tmp > error)
        error = tmp*tmp;
    }
    MPI_Reduce(&error, &error_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    pfft_printf(comm, "%s, %d. component - absolute rms error = %6.2e,  relative rms error =  %6.2e\n", name, t, sqrt(error_max), sqrt(error_max/sum));
  }
}

