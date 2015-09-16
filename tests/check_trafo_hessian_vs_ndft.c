#include <stdlib.h>
#include <complex.h>
#include <pnfft.h>

static void pnfft_perform_guru(
    const ptrdiff_t *N, const ptrdiff_t *n, ptrdiff_t local_M,
    int m, const double *x_max, unsigned pnfft_flags,
    const int *np, MPI_Comm comm,
    int debug);

static void init_parameters(
    int argc, char **argv,
    ptrdiff_t *N, ptrdiff_t *n, ptrdiff_t *M,
    int *m, int *window, int *intpol, int *interlacing, int *diff_ik,
    double *x_max, int *np, int *debug);
static void init_random_x(
    const double *lo, const double *up,
    const double *x_max, ptrdiff_t M,
    double *x);
static void compare_f(
    const pnfft_complex *f1, const pnfft_complex *f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm);
static void compare_hessian_f(
    const pnfft_complex *hessian_f1, const pnfft_complex *hessian_f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm);
static double random_number_less_than_one(
    void);


int main(int argc, char **argv){
  int np[3], m, window, interlacing, diff_ik, debug;
  ptrdiff_t N[3], n[3], local_M;
  double x_max[3];
  
  MPI_Init(&argc, &argv);
  pnfft_init();
  
  /* default values */
  N[0] = N[1] = N[2] = 16;
  n[0] = n[1] = n[2] = 0;
  local_M = 0;
  m = 6;
  window = 4;
  interlacing = 0;
  diff_ik = 0;
  x_max[0] = x_max[1] = x_max[2] = 0.5;
  np[0]=2; np[1]=2; np[2]=2;
  debug=0;
  
  /* set values by commandline */
  int intpol = -1;
  init_parameters(argc, argv, N, n, &local_M, &m, &window, &intpol, &interlacing, &diff_ik, x_max, np, &debug);

  /* if M or n are set to zero, we choose nice values */
  local_M = (local_M==0) ? N[0]*N[1]*N[2]/(np[0]*np[1]*np[2]) : local_M;
  for(int t=0; t<3; t++)
    n[t] = (n[t]==0) ? 2*N[t] : n[t];

  unsigned window_flag;
  switch(window){
    case 0: window_flag = PNFFT_WINDOW_GAUSSIAN; break;
    case 1: window_flag = PNFFT_WINDOW_BSPLINE; break;
    case 2: window_flag = PNFFT_WINDOW_SINC_POWER; break;
    case 3: window_flag = PNFFT_WINDOW_BESSEL_I0; break;
    default: window_flag = PNFFT_WINDOW_KAISER_BESSEL;
  }

  unsigned intpol_flag;
  switch(intpol){
    case 0: intpol_flag = PNFFT_PRE_CONST_PSI; break;
    case 1: intpol_flag = PNFFT_PRE_LIN_PSI; break;
    case 2: intpol_flag = PNFFT_PRE_QUAD_PSI; break;
    case 3: intpol_flag = PNFFT_PRE_CUB_PSI; break;
    default: intpol_flag = (window==0) ? PNFFT_FAST_GAUSSIAN : 0;
  }

  unsigned interlacing_flag = (interlacing) ? PNFFT_INTERLACED : 0;
  unsigned diff_ik_flag     = (diff_ik)  ? PNFFT_DIFF_IK : PNFFT_DIFF_AD;

  pfft_printf(MPI_COMM_WORLD, "******************************************************************************************************\n");
  pfft_printf(MPI_COMM_WORLD, "* Computation of parallel NFFT\n");
  pfft_printf(MPI_COMM_WORLD, "* for  N[0] x N[1] x N[2] = %td x %td x %td Fourier coefficients (change with -pnfft_N * * *)\n", N[0], N[1], N[2]);
  pfft_printf(MPI_COMM_WORLD, "* at   local_M = %td nodes per process (change with -pnfft_local_M *)\n", local_M);
  pfft_printf(MPI_COMM_WORLD, "* with n[0] x n[1] x n[2] = %td x %td x %td FFT grid size (change with -pnfft_n * * *),\n", n[0], n[1], n[2]);
  pfft_printf(MPI_COMM_WORLD, "*      m = %td real space cutoff (change with -pnfft_m *),\n", m);
  pfft_printf(MPI_COMM_WORLD, "*      window = %d window function ", window);
  switch(window){
    case 0: pfft_printf(MPI_COMM_WORLD, "(PNFFT_WINDOW_GAUSSIAN) "); break;
    case 1: pfft_printf(MPI_COMM_WORLD, "(PNFFT_WINDOW_BSPLINE) "); break;
    case 2: pfft_printf(MPI_COMM_WORLD, "(PNFFT_WINDOW_SINC_POWER) "); break;
    case 3: pfft_printf(MPI_COMM_WORLD, "(PNFFT_WINDOW_BESSEL_I0) "); break;
    default: pfft_printf(MPI_COMM_WORLD, "(PNFFT_WINDOW_KAISER_BESSEL) ");
  }
  pfft_printf(MPI_COMM_WORLD, "(change with -pnfft_window *),\n");
  pfft_printf(MPI_COMM_WORLD, "*      intpol = %d interpolation order ", intpol);
  switch(intpol){
    case 0: pfft_printf(MPI_COMM_WORLD, "(PNFFT_PRE_CONST_PSI) "); break;
    case 1: pfft_printf(MPI_COMM_WORLD, "(PNFFT_PRE_LIN_PSI) "); break;
    case 2: pfft_printf(MPI_COMM_WORLD, "(PNFFT_PRE_QUAD_PSI) "); break;
    case 3: pfft_printf(MPI_COMM_WORLD, "(PNFFT_PRE_CUB_PSI) "); break;
    default: if(window==0)
               pfft_printf(MPI_COMM_WORLD, "(PNFFT_FAST_GAUSSIAN) ");
             else
               pfft_printf(MPI_COMM_WORLD, "(No interpolation enabled) ");
  }
  pfft_printf(MPI_COMM_WORLD, "(change with -pnfft_intpol *),\n");
  if(interlacing)
    pfft_printf(MPI_COMM_WORLD, "*      interlacing = enabled (disable with -pnfft_interlacing 0)\n");
  else
    pfft_printf(MPI_COMM_WORLD, "*      interlacing = disabled (enable with -pnfft_interlacing 1)\n");
  if(diff_ik)
    pfft_printf(MPI_COMM_WORLD, "*      derivative = diff-ik (enable diff-ad with -pnfft_diff_ik 0)\n");
  else
    pfft_printf(MPI_COMM_WORLD, "*      derivative = diff-ad (enable diff-ik with -pnfft_diff_ik 1)\n");
  pfft_printf(MPI_COMM_WORLD, "* on   np[0] x np[1] x np[2] = %td x %td x %td processes (change with -pnfft_np * * *)\n", np[0], np[1], np[2]);
  pfft_printf(MPI_COMM_WORLD, "*******************************************************************************************************\n\n");

//  window_flag |= PNFFT_PRE_CUB_PSI;

  /* calculate parallel NFFT */
  pnfft_perform_guru(N, n, local_M, m,   x_max, window_flag| intpol_flag| interlacing_flag| diff_ik_flag, np, MPI_COMM_WORLD, debug);

  /* free mem and finalize */
  pnfft_cleanup();
  MPI_Finalize();
  return 0;
}


static void pnfft_perform_guru(
    const ptrdiff_t *N, const ptrdiff_t *n, ptrdiff_t local_M,
    int m, const double *x_max, unsigned pnfft_flags,
    const int *np, MPI_Comm comm,
    int debug
    )
{
  int myrank;
  ptrdiff_t local_N[3], local_N_start[3];
  double lower_border[3], upper_border[3];
  double local_sum = 0, time, time_max;
  MPI_Comm comm_cart_3d;
  pnfft_complex *f_hat, *f, *f1, *hessian_f, *hessian_f1;
  double *x, f_hat_sum;
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
  pnfft_local_size_guru(3, N, n, x_max, m, comm_cart_3d, PNFFT_TRANSPOSED_NONE,
      local_N, local_N_start, lower_border, upper_border);

  /* plan parallel NFFT */
  pnfft = pnfft_init_guru(3, N, n, x_max, m,
      PNFFT_MALLOC_F_HAT | pnfft_flags, PFFT_ESTIMATE,
      comm_cart_3d);

  /* initialize nodes */
  nodes = pnfft_init_nodes(local_M, PNFFT_MALLOC_X | PNFFT_MALLOC_F | PNFFT_MALLOC_HESSIAN_F);

  /* get data pointers */
  f_hat     = pnfft_get_f_hat(pnfft);
  f         = pnfft_get_f(nodes);
  hessian_f = pnfft_get_hessian_f(nodes);
  x         = pnfft_get_x(nodes);

  /* initialize Fourier coefficients with random numbers */
  pnfft_init_f_hat_3d(N, local_N, local_N_start, PNFFT_TRANSPOSED_NONE,
      f_hat);
  
  if(debug){
    /* debug mode does NOT work with parallel data distribution */
    if(np[0]*np[1]*np[2] > 2){
      fprintf(stderr, "Error: debugging mode is only valid for 1 and 2 core runs !!!\n");
      exit(1);
    }

    ptrdiff_t l=0;
    for(ptrdiff_t k0=local_N_start[0]; k0<local_N_start[0] + local_N[0]; k0++)
      for(ptrdiff_t k1=local_N_start[1]; k1<local_N_start[1] + local_N[1]; k1++)
        for(ptrdiff_t k2=local_N_start[2]; k2<local_N_start[2] + local_N[2]; k2++, l++)
          f_hat[l] = sqrt(l+1.0) - 3.0/(l+1.0) * I;

    /* print Matlab-like */
    for(int p=0; p<np[0]*np[1]*np[2]; p++){
      if(myrank==p){
        for(ptrdiff_t k2=local_N_start[2]; k2<local_N_start[2] + local_N[2]; k2++){
          for(ptrdiff_t k0=local_N_start[0]; k0<local_N_start[0] + local_N[0]; k0++){
            for(ptrdiff_t k1=local_N_start[1]; k1<local_N_start[1] + local_N[1]; k1++){
              ptrdiff_t l0 = k0-local_N_start[0];
              ptrdiff_t l1 = k1-local_N_start[1];
              ptrdiff_t l2 = k2-local_N_start[2];
              l = l2 + l1 * local_N[2] + l0 * local_N[1]*local_N[2];
              fprintf(stderr, "f_hat[%td, %td, %td] = %f + %fi,   ", k0+N[0]/2+1, k1+N[1]/2+1, k2+N[2]/2+1, creal(f_hat[l]), cimag(f_hat[l]));
            }
            fprintf(stderr, "\n");
          }
          fprintf(stderr, "\n");
        }
      }
      MPI_Barrier(comm_cart_3d);
    }
  }

  /* initialize nodes with random numbers */
  srand(myrank);
  init_random_x(lower_border, upper_border, x_max, local_M,
      x);
  
  if(debug){
    MPI_Barrier(comm_cart_3d);
    double shift=0.1;
    for(ptrdiff_t j=0; j<local_M; j++)
      for(int t=0; t<3; t++)
        x[3*j+t] = ( (double)j/local_M + t*shift) - 0.5 * (myrank==0);
    for(ptrdiff_t j=0; j<local_M; j++)
      fprintf(stderr, "x(%td) = [%.2e, %.2e, %.2e]\n", j, x[3*j], x[3*j+1], x[3*j+2]);
  }
    
  /* execute parallel NFFT */
  time = -MPI_Wtime();
  pnfft_trafo(pnfft, nodes, PNFFT_COMPUTE_F | PNFFT_COMPUTE_HESSIAN_F);
  time += MPI_Wtime();

  if(debug){
    for(int p=0; p<np[0]*np[1]*np[2]; p++){
      if(myrank==p){
        for(ptrdiff_t j=0; j<local_M; j++){
          fprintf(stderr, "pnfft hessian(%d, %td) = [ ", myrank, j);
          for(int t=0; t<6; t++)
            fprintf(stderr, "%.2e + %.2e * I,   ", creal(hessian_f[6*j+t]), cimag(hessian_f[6*j+t]));
          fprintf(stderr, "]\n");
        }
      }
      MPI_Barrier(comm_cart_3d);
    }
  }
  
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "pnfft_trafo with Hessian needs %6.2e s\n", time_max);
 
  /* calculate norm of Fourier coefficients for calculation of relative error */ 
  for(ptrdiff_t k=0; k<local_N[0]*local_N[1]*local_N[2]; k++)
    local_sum += cabs(f_hat[k]);
  MPI_Allreduce(&local_sum, &f_hat_sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart_3d);

  /* store results of NFFT */
  f1 = pnfft_alloc_complex(local_M);
  for(ptrdiff_t j=0; j<local_M; j++) f1[j] = f[j];

  hessian_f1 = pnfft_alloc_complex(6*local_M);
  for(ptrdiff_t j=0; j<6*local_M; j++) hessian_f1[j] = hessian_f[j];

  /* execute parallel NDFT */
  time = -MPI_Wtime();
  pnfft_trafo(pnfft, nodes, PNFFT_COMPUTE_DIRECT | PNFFT_COMPUTE_F | PNFFT_COMPUTE_HESSIAN_F);
  time += MPI_Wtime();

  if(debug){
    for(int p=0; p<np[0]*np[1]*np[2]; p++){
      if(myrank==p){
        for(ptrdiff_t j=0; j<local_M; j++){
          fprintf(stderr, "pndft hessian(%d, %td) = [ ", myrank, j);
          for(int t=0; t<6; t++)
            fprintf(stderr, "%.2e + %.2e * I,   ", creal(hessian_f[6*j+t]), cimag(hessian_f[6*j+t]));
          fprintf(stderr, "]\n");
        }
      }
      MPI_Barrier(comm_cart_3d);
    }
  }

  /* print timing */
  MPI_Reduce(&time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "direct pnfft_trafo with Hessian needs %6.2e s\n", time_max);

  /* calculate error of PNFFT */
  compare_f(f1, f, local_M, f_hat_sum, "* Results in f", MPI_COMM_WORLD);
  compare_hessian_f(hessian_f1, hessian_f, local_M, f_hat_sum, "* Results in hessian_f", MPI_COMM_WORLD);

  /* free mem and finalize */
  pnfft_free(f1); pnfft_free(hessian_f1);
  pnfft_finalize(pnfft, PNFFT_FREE_F_HAT);
  pnfft_free_nodes(nodes, PNFFT_FREE_X | PNFFT_FREE_F | PNFFT_FREE_HESSIAN_F);
  MPI_Comm_free(&comm_cart_3d);
}


static void init_parameters(
    int argc, char **argv,
    ptrdiff_t *N, ptrdiff_t *n, ptrdiff_t *M,
    int *m, int *window, int *intpol, int *interlacing, int *diff_ik,
    double *x_max, int *np, int *debug
    )
{
  pfft_get_args(argc, argv, "-pnfft_local_M", 1, PFFT_PTRDIFF_T, M);
  pfft_get_args(argc, argv, "-pnfft_N", 3, PFFT_PTRDIFF_T, N);
  pfft_get_args(argc, argv, "-pnfft_n", 3, PFFT_PTRDIFF_T, n);
  pfft_get_args(argc, argv, "-pnfft_np", 3, PFFT_INT, np);
  pfft_get_args(argc, argv, "-pnfft_m", 1, PFFT_INT, m);
  pfft_get_args(argc, argv, "-pnfft_window", 1, PFFT_INT, window);
  pfft_get_args(argc, argv, "-pnfft_intpol", 1, PFFT_INT, intpol);
  pfft_get_args(argc, argv, "-pnfft_interlacing", 1, PFFT_INT, interlacing);
  pfft_get_args(argc, argv, "-pnfft_diff_ik", 1, PFFT_INT, diff_ik);
  pfft_get_args(argc, argv, "-pnfft_x_max", 3, PFFT_DOUBLE, x_max);
  pfft_get_args(argc, argv, "-pnfft_debug", 1, PFFT_INT, debug);
}


static void compare_f(
    const pnfft_complex *f1, const pnfft_complex *f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm
    )
{
  double error = 0, error_max;

  for(ptrdiff_t j=0; j<local_M; j++)
    if( cabs(f1[j]-f2[j]) > error)
      error = cabs(f1[j]-f2[j]);

  MPI_Reduce(&error, &error_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  pfft_printf(comm, "%s absolute error = %6.2e\n", name, error_max);
  pfft_printf(comm, "%s relative error = %6.2e\n", name, error_max/f_hat_sum);
}

static void compare_hessian_f(
    const pnfft_complex *hessian_f1, const pnfft_complex *hessian_f2, ptrdiff_t local_M,
    double f_hat_sum, const char *name, MPI_Comm comm
    )
{
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

static void init_random_x(
    const double *lo, const double *up,
    const double *x_max, ptrdiff_t M,
    double *x
    )
{
  double tmp;
  
  for (ptrdiff_t j=0; j<M; j++){
    for(int t=0; t<3; t++){
      do{
        tmp = random_number_less_than_one();
        tmp = (up[t]-lo[t]) * tmp + lo[t];
      }
      while( (tmp < -x_max[t]) || (x_max[t] <= tmp) );
      x[3*j+t] = tmp;
    }
  }
}


static double random_number_less_than_one(
    void
    )
{
  double tmp;
  
  do
    tmp = ( 1.0 * rand()) / RAND_MAX;
  while(tmp>=1.0);
  
  return tmp;
}

