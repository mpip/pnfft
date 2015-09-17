/*
 * Copyright (c) 2011-2013 Michael Pippig
 *
 * This file is part of PNFFT.
 *
 * PNFFT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PNFFT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PNFFT.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <complex.h>
#include "pnfft.h"
#include "ipnfft.h"
#include "matrix_D.h"

/* wrappers for pfft init and cleanup */
void PNX(init) (void){
  PX(init)();
}


void PNX(cleanup) (void){
  PX(cleanup)();
}

void PNX(local_size_3d)(
    const INT *N, MPI_Comm comm_cart,
    unsigned pnfft_flags,
    INT *local_N, INT *local_N_start,
    R *lower_border, R *upper_border
    )
{
  const int d=3;

  PNX(local_size_adv)(
      d, N, comm_cart, pnfft_flags,
      local_N, local_N_start, lower_border, upper_border);
}


void PNX(local_size_3d_c2r)(
    const INT *N, MPI_Comm comm_cart,
    unsigned pnfft_flags,
    INT *local_N, INT *local_N_start,
    R *lower_border, R *upper_border
    )
{
  const int d=3;

  PNX(local_size_adv_c2r)(
      d, N, comm_cart, pnfft_flags,
      local_N, local_N_start, lower_border, upper_border);
}


PNX(plan) PNX(init_3d)(
    const INT *N, 
    MPI_Comm comm_cart
    )
{
  const int d=3;
  unsigned pnfft_flags, pfft_flags;

  pnfft_flags = PNFFT_MALLOC_F_HAT;
  pfft_flags = PFFT_MEASURE| PFFT_DESTROY_INPUT;

  return PNX(init_adv)(
      d, N,
      pnfft_flags, pfft_flags, comm_cart);
}


PNX(plan) PNX(init_3d_c2r)(
    const INT *N, 
    MPI_Comm comm_cart
    )
{
  const int d=3;
  unsigned pnfft_flags, pfft_flags;

  pnfft_flags = PNFFT_MALLOC_F_HAT;
  pfft_flags = PFFT_MEASURE| PFFT_DESTROY_INPUT;

  return PNX(init_adv_c2r)(
      d, N,
      pnfft_flags, pfft_flags, comm_cart);
}


static void trafo_F_and_B_ik_complex_input(
    PNX(plan) ths, PNX(nodes) nodes, int use_interlacing, int interlaced, unsigned compute_flags
    )
{
  /* duplicate g1 since we have to scale it several times for computing gradient/Hessian */
  if( compute_flags & (PNFFT_COMPUTE_GRAD_F | PNFFT_COMPUTE_HESSIAN_F) ){
    PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_D]);
    if( ~compute_flags & PNFFT_OMIT_DECONV )
      for(INT k=0; k<ths->local_N_total; k++)
        ((C*)ths->g1_buffer)[k] = ((C*)ths->g1)[k];
    PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_D]);
  }

  /* calculate potentials */
  if( compute_flags & PNFFT_COMPUTE_F){
    PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_F]);
    if( ~compute_flags & PNFFT_OMIT_FFT )
      PNX(trafo_F)(ths);
    PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_F]);

    PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);
    if( ~compute_flags & PNFFT_OMIT_CONV )
      PNX(trafo_B_ad)(ths, nodes, nodes->f, NULL, NULL, 0, 1, use_interlacing, interlaced, PNFFT_COMPUTE_F);
    PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);
  }

  /* calculate gradient component wise */
  if(compute_flags & PNFFT_COMPUTE_GRAD_F){
    for(int dim =0; dim<3; dim++){
      PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_D]);
      if( ~compute_flags & PNFFT_OMIT_DECONV )
        PNX(trafo_scale_ik_diff_c2c)((C*)ths->g1_buffer, ths->local_N_start, ths->local_N, dim, ths->pnfft_flags,
            (C*)ths->g1);
      PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_D]);
      
      PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_F]);
      if( ~compute_flags & PNFFT_OMIT_FFT )
        PNX(trafo_F)(ths);
      PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_F]);

      PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);
      if( ~compute_flags & PNFFT_OMIT_CONV )
        PNX(trafo_B_ad)(ths, nodes, nodes->grad_f, NULL, NULL, dim, 3, use_interlacing, interlaced, PNFFT_COMPUTE_F);
      PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);
    }
  }

  /* calculate Hessian component wise */
  if(compute_flags & PNFFT_COMPUTE_HESSIAN_F){
    for(int dim =0; dim<6; dim++){
      PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_D]);
      if( ~compute_flags & PNFFT_OMIT_DECONV )
        PNX(trafo_scale_ik_diff2_c2c)((C*)ths->g1_buffer, ths->local_N_start, ths->local_N, dim, ths->pnfft_flags,
            (C*)ths->g1);
      PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_D]);
      
      PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_F]);
      if( ~compute_flags & PNFFT_OMIT_FFT )
        PNX(trafo_F)(ths);
      PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_F]);

      PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);
      if( ~compute_flags & PNFFT_OMIT_CONV )
        PNX(trafo_B_ad)(ths, nodes, nodes->hessian_f, NULL, NULL, dim, 6, use_interlacing, interlaced, PNFFT_COMPUTE_F);
      PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);
    }
  }
}


static void trafo(
    PNX(plan) ths, PNX(nodes) nodes, int use_interlacing, int interlaced, unsigned compute_flags
    )
{
  /* multiplication with matrix D */
  PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_D]);
  if( ~compute_flags & PNFFT_OMIT_DECONV )
    PNX(trafo_D)(ths, interlaced);
  PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_D]);
 
  if( ths->pnfft_flags & PNFFT_DIFF_IK ){
    /* multiplication with matrix F and B for ik-differentiation */
    trafo_F_and_B_ik_complex_input(ths, nodes, use_interlacing, interlaced, compute_flags);
  } else {
    /* multiplication with matrix F */
    PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_F]);
    if( ~compute_flags & PNFFT_OMIT_FFT )
      PNX(trafo_F)(ths);
    PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_F]);

    /* multiplication with matrix B */
    PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);
    if( ~compute_flags & PNFFT_OMIT_CONV )
      PNX(trafo_B_ad)(ths, nodes, nodes->f, nodes->grad_f, nodes->hessian_f, 0, 1, use_interlacing, interlaced, compute_flags);
    PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);
  }
}

/* parallel 3dNFFT with different window functions */
void PNX(trafo)(
    PNX(plan) ths, PNX(nodes) nodes, unsigned compute_flags
    )
{
  int use_interlacing, interlaced;

  if(ths==NULL){
    PX(fprintf)(MPI_COMM_WORLD, stderr, "!!! Error: Can not execute PNFFT Plan == NULL !!!\n");
    return;
  }

  if(compute_flags & PNFFT_COMPUTE_DIRECT){
    PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_WHOLE]);

    PNX(trafo_A)(ths, nodes, compute_flags);
    
    ths->timer_trafo[PNFFT_TIMER_ITER]++;
    PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_WHOLE]);
    return;
  }

  PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_WHOLE]);

  PNFFT_START_TIMING(ths->comm_cart, ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);
  if( ~compute_flags & PNFFT_COMPUTE_ACCUMULATED ){
    INT tuple = (ths->trafo_flag & PNFFTI_TRAFO_C2R) ? 1 : 2;

    if(compute_flags & PNFFT_COMPUTE_F)
      for(INT m=0; m<tuple*nodes->local_M; m++)
        nodes->f[m] = 0;
    if(compute_flags & PNFFT_COMPUTE_GRAD_F)
      for(INT m=0; m<3*tuple*nodes->local_M; m++)
        nodes->grad_f[m] = 0;
    if(compute_flags & PNFFT_COMPUTE_HESSIAN_F)
      for(INT m=0; m<6*tuple*nodes->local_M; m++)
        nodes->hessian_f[m] = 0;
  }
  PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_MATRIX_B]);

  if(ths->pnfft_flags & PNFFT_INTERLACED){
    /* compute interlaced NFFT and average the results */
    trafo(ths, nodes, use_interlacing=1, interlaced=0, compute_flags);
    trafo(ths, nodes, use_interlacing=1, interlaced=1, compute_flags);
  } else {
    /* compute non-interlaced NFFT */
    trafo(ths, nodes, use_interlacing=0, interlaced=0, compute_flags);
  }     
 
  ths->timer_trafo[PNFFT_TIMER_ITER]++;
  PNFFT_FINISH_TIMING(ths->timer_trafo[PNFFT_TIMER_WHOLE]);
}




static void adjoint_B_and_F_ik_complex_input(
    PNX(plan) ths, PNX(nodes) nodes, int use_interlacing, int interlaced, unsigned compute_flags
    )
{
  /* save g1 since we want to accumulate all results at the end */
  PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_B]);
  if( ~compute_flags & PNFFT_OMIT_DECONV )
    for(INT k=0; k<ths->local_N_total; k++)
      ((C*)ths->g1_buffer)[k] = 0;
//       ((C*)ths->g1_buffer)[k] = ((C*)ths->g1)[k];
  PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_B]);

  /* spread potentials */
  if( compute_flags & PNFFT_COMPUTE_F){
    PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_B]);
    if( ~compute_flags & PNFFT_OMIT_CONV )
      PNX(adjoint_B_ad)(ths, nodes, nodes->f, NULL, 0, 1, use_interlacing, interlaced, PNFFT_COMPUTE_F);
    PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_B]);

    PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_F]);
    if( ~compute_flags & PNFFT_OMIT_FFT )
      PNX(adjoint_F)(ths);
    PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_F]);

    PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_D]);
    for(INT k=0; k<ths->local_N_total; k++)
      ((C*)ths->g1_buffer)[k] += ((C*)ths->g1)[k];
    PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_D]);
  }

  /* spread gradient component wise */
  if(compute_flags & PNFFT_COMPUTE_GRAD_F){
    for(int dim =0; dim<3; dim++){
      PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_B]);
      if( ~compute_flags & PNFFT_OMIT_CONV ){
        PNX(adjoint_B_ad)(ths, nodes, nodes->grad_f, NULL, dim, 3, use_interlacing, interlaced, PNFFT_COMPUTE_F);
      }
      PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_B]);
      
      PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_F]);
      if( ~compute_flags & PNFFT_OMIT_FFT )
        PNX(adjoint_F)(ths);
      PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_F]);

      PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_D]);
      if( ~compute_flags & PNFFT_OMIT_DECONV )
        PNX(adjoint_scale_ik_diff_c2c)((C*)ths->g1, ths->local_N_start, ths->local_N, dim, ths->pnfft_flags,
            (C*)ths->g1_buffer);
      PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_D]);
    }
  }

  PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_D]);
  for(INT k=0; k<ths->local_N_total; k++)
    ((C*)ths->g1)[k] = ((C*)ths->g1_buffer)[k];
  PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_D]);
}

static void adj(
    PNX(plan) ths, PNX(nodes) nodes, int use_interlacing, int interlaced, unsigned compute_flags
    )
{
  if( ths->pnfft_flags & PNFFT_DIFF_IK ){
    /* multiplication with matrix B^T and F^H for ik-differentiation */
    adjoint_B_and_F_ik_complex_input(ths, nodes, use_interlacing, interlaced, compute_flags);
  } else {
    /* multiplication with matrix B^T */
    PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_B]);
    if( ~compute_flags & PNFFT_OMIT_CONV )
      PNX(adjoint_B_ad)(ths, nodes, nodes->f, nodes->grad_f, 0, 1, use_interlacing, interlaced, compute_flags);
    PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_B]);

    /* multiplication with matrix F^H */
    PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_F]);
    if( ~compute_flags & PNFFT_OMIT_FFT )
      PNX(adjoint_F)(ths);
    PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_F]);
  }

  /* multiplication with matrix D */
  PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_MATRIX_D]);
  if( ~compute_flags & PNFFT_OMIT_DECONV )
    PNX(adjoint_D)(ths, interlaced);
  PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_MATRIX_D]);
}

void PNX(adj)(
    PNX(plan) ths, PNX(nodes) nodes, unsigned compute_flags
    )
{
  int use_interlacing, interlaced;

  if(ths==NULL){
    PX(fprintf)(MPI_COMM_WORLD, stderr, "!!! Error: Can not execute PNFFT Plan == NULL !!!\n");
    return;
  }

  if(compute_flags & PNFFT_COMPUTE_DIRECT){
    PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_WHOLE]);

    PNX(adj_A)(ths, nodes, compute_flags);

    ths->timer_adj[PNFFT_TIMER_ITER]++;
    PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_WHOLE]);
    return;
  }

  PNFFT_START_TIMING(ths->comm_cart, ths->timer_adj[PNFFT_TIMER_WHOLE]);

  if( ~compute_flags & PNFFT_COMPUTE_ACCUMULATED )
    for(INT m=0; m<ths->local_N_total; m++)
      ths->f_hat[m] = 0;

  if(ths->pnfft_flags & PNFFT_INTERLACED){
    /* compute interlaced NFFT and average the results */
    adj(ths, nodes, use_interlacing=1, interlaced=0, compute_flags);
    adj(ths, nodes, use_interlacing=1, interlaced=1, compute_flags);
  } else {
    /* compute non-interlaced NFFT */
    adj(ths, nodes, use_interlacing=0, interlaced=0, compute_flags);
  }

  ths->timer_adj[PNFFT_TIMER_ITER]++;
  PNFFT_FINISH_TIMING(ths->timer_adj[PNFFT_TIMER_WHOLE]);
}

static PNX(nodes) mknodes(
    void
    )
{
  PNX(nodes) nodes = (nodes_s*) malloc(sizeof(nodes_s));

  nodes->local_M   = 0;
  nodes->f         = NULL;
  nodes->grad_f    = NULL;
  nodes->hessian_f = NULL;
  nodes->x         = NULL;

  nodes->pre_psi      = NULL;
  nodes->pre_dpsi     = NULL;
  nodes->pre_ddpsi    = NULL;
  nodes->pre_psi_il   = NULL;
  nodes->pre_dpsi_il  = NULL;
  nodes->pre_ddpsi_il = NULL;

  return nodes;
}

PNX(nodes) PNX(init_nodes)(
    INT local_M, unsigned malloc_flags
    )
{
  PNX(nodes) nodes = mknodes();

  nodes->local_M = local_M;
  nodes->precompute_flags = 0;

  /* allocate mem */
  PNX(malloc_x)(nodes, malloc_flags);
  PNX(malloc_f)(nodes, malloc_flags);
  PNX(malloc_grad_f)(nodes, malloc_flags);
  PNX(malloc_hessian_f)(nodes, malloc_flags);

  return nodes;
}

void PNX(free_nodes)(
    PNX(nodes) nodes, unsigned pnfft_finalize_flags
    )
{
  /* plan was already destroyed or never initialized */
  if(nodes==NULL)
    return;

  if((pnfft_finalize_flags & PNFFT_FREE_GRAD_F) && (nodes->grad_f != NULL))
    PNX(free)(nodes->grad_f);
  if((pnfft_finalize_flags & PNFFT_FREE_HESSIAN_F) && (nodes->hessian_f != NULL))
    PNX(free)(nodes->hessian_f);
  if((pnfft_finalize_flags & PNFFT_FREE_F) && (nodes->f != NULL))
    PNX(free)(nodes->f);
  if((pnfft_finalize_flags & PNFFT_FREE_X) && (nodes->x != NULL))
    PNX(free)(nodes->x);

  if(nodes->pre_psi != NULL)      PNX(free)(nodes->pre_psi);
  if(nodes->pre_dpsi != NULL)     PNX(free)(nodes->pre_dpsi);
  if(nodes->pre_ddpsi != NULL)    PNX(free)(nodes->pre_ddpsi);
  if(nodes->pre_psi_il != NULL)   PNX(free)(nodes->pre_psi_il);
  if(nodes->pre_dpsi_il != NULL)  PNX(free)(nodes->pre_dpsi_il);
  if(nodes->pre_ddpsi_il != NULL) PNX(free)(nodes->pre_ddpsi_il);

  /* free memory */
  free(nodes);
  /* nodes=NULL; would be senseless, since we can not change the pointer itself */
}


void PNX(finalize)(
    PNX(plan) ths, unsigned pnfft_finalize_flags
    )
{
  if((pnfft_finalize_flags & PNFFT_FREE_F_HAT) && (ths->f_hat != NULL))
    PNX(free)(ths->f_hat);

  PX(destroy_plan)(ths->pfft_forw);
  PX(destroy_plan)(ths->pfft_back);
  PX(destroy_gcplan)(ths->gcplan);

  if(ths->g2 != ths->g1){
    if(ths->g2 != NULL)
      PNX(free)(ths->g2);
  }
  if(ths->g1 != NULL)
    PNX(free)(ths->g1);
  if(ths->g1_buffer != NULL)
    PNX(free)(ths->g1_buffer);

  if(ths->intpol_tables_psi != NULL){
    for(int t=0;t<ths->d; t++)
      if(ths->intpol_tables_psi[t] != NULL)
        PNX(free)(ths->intpol_tables_psi[t]);
    PNX(free)(ths->intpol_tables_psi);
  }

  if(ths->intpol_tables_dpsi != NULL){
    for(int t=0;t<ths->d; t++)
      if(ths->intpol_tables_dpsi[t] != NULL)
        PNX(free)(ths->intpol_tables_dpsi[t]);
    PNX(free)(ths->intpol_tables_dpsi);
  }

  PNX(free)(ths->x_max);
  PNX(free)(ths->sigma);
  PNX(free)(ths->n);
  PNX(free)(ths->no);
  PNX(free)(ths->N);
  PNX(free)(ths->local_N);
  PNX(free)(ths->local_N_start);
  PNX(free)(ths->local_no);
  PNX(free)(ths->local_no_start);

  MPI_Comm_free(&(ths->comm_cart));

  /* finalize window specific parameters */
  if(ths->b != NULL)
    PNX(free)(ths->b);
  if(ths->exp_const != NULL)
    PNX(free)(ths->exp_const);
  if(ths->spline_coeffs != NULL)
    PNX(free)(ths->spline_coeffs);
  if(ths->pre_inv_phi_hat_trafo != NULL)
    PNX(free)(ths->pre_inv_phi_hat_trafo);
  if(ths->pre_inv_phi_hat_adj != NULL)
    PNX(free)(ths->pre_inv_phi_hat_adj);

  /* free mem of struct */
  PNX(rmplan)(ths);
}

void PNX(set_f_hat)(
    C *f_hat, PNX(plan) ths
    )
{
  ths->f_hat = f_hat;
}

C* PNX(get_f_hat)(
    const PNX(plan) ths
    )
{
  return ths->f_hat;
}

void PNX(set_f)(
    C *f, PNX(nodes) nodes
    )
{
  nodes->f = (R*)f;
}

C* PNX(get_f)(
    const PNX(nodes) nodes
    )
{
  return (C*)nodes->f;
}

void PNX(set_grad_f)(
    C* grad_f, PNX(nodes) nodes
    )
{
  nodes->grad_f = (R*)grad_f;
}

C* PNX(get_grad_f)(
    const PNX(nodes) nodes
    )
{
  return (C*)nodes->grad_f;
}

void PNX(set_hessian_f)(
    C* hessian_f, PNX(nodes) nodes
    )
{
  nodes->hessian_f = (R*)hessian_f;
}

C* PNX(get_hessian_f)(
    const PNX(nodes) nodes
    )
{
  return (C*)nodes->hessian_f;
}

void PNX(set_f_hat_real)(
    R *f_hat, PNX(plan) ths
    )
{
  ths->f_hat = (C*)f_hat;
}

R* PNX(get_f_hat_real)(
    const PNX(plan) ths
    )
{
  return (R*)ths->f_hat;
}

void PNX(set_f_real)(
    R *f, PNX(nodes) nodes
    )
{
  nodes->f = f;
}

R* PNX(get_f_real)(
    const PNX(nodes) nodes
    )
{
  return nodes->f;
}

void PNX(set_grad_f_real)(
    R* grad_f, PNX(nodes) nodes
    )
{
  nodes->grad_f = grad_f;
}

R* PNX(get_grad_f_real)(
    const PNX(nodes) nodes
    )
{
  return nodes->grad_f;
}

void PNX(set_hessian_f_real)(
    R* hessian_f, PNX(nodes) nodes
    )
{
  nodes->hessian_f = hessian_f;
}

R* PNX(get_hessian_f_real)(
    const PNX(nodes) nodes
    )
{
  return nodes->hessian_f;
}

void PNX(set_x)(
    R *x, PNX(nodes) nodes
    )
{
  nodes->x = x;
}

R* PNX(get_x)(
    const PNX(nodes) nodes
    )
{
  return nodes->x;
}


/* getters for PNFFT internal parameters
 * No setters are implemented for these parameters.
 * Use finalize and init_guru instead. */
void PNX(set_b)(
    R b0, R b1, R b2,
    PNX(plan) ths
    )
{
  ths->b[0] = b0;
  ths->b[1] = b1;
  ths->b[2] = b2;
  PNX(init_precompute_window)(ths);
}

void PNX(get_b)(
    const PNX(plan) ths,
    R *b0, R *b1, R *b2
    )
{
  *b0 = ths->b[0];
  *b1 = ths->b[1];
  *b2 = ths->b[2];
}

int PNX(get_d)(
    const PNX(plan) ths
    )
{
  return ths->d;
}

int PNX(get_m)(
    const PNX(plan) ths
    )
{
  return ths->m;
}

void PNX(get_x_max)(
    const PNX(plan) ths,
    R *x_max
    )
{
  for(int t=0; t<ths->d; t++)
    x_max[t] = ths->x_max[t];
}

void PNX(get_N)(
    const PNX(plan) ths,
    INT *N
    )
{
  for(int t=0; t<ths->d; t++)
    N[t] = ths->N[t];
}

void PNX(get_n)(
    const PNX(plan) ths,
    INT *n
    )
{
  for(int t=0; t<ths->d; t++)
    n[t] = ths->n[t];
}

unsigned PNX(get_pnfft_flags)(
    const PNX(plan) ths
    )
{
  return ths->pnfft_flags;
}

unsigned PNX(get_pfft_flags)(
    const PNX(plan) ths
    )
{
  return ths->pfft_opt_flags;
}

void PNX(init_f_hat_3d)(
    const INT *N, const INT *local_N, const INT *local_N_start,
    unsigned pnfft_flags,
    C *data
    )
{
  INT local_Nt[3], local_Nt_start[3];
  int shift = (pnfft_flags & PNFFT_TRANSPOSED_F_HAT) ? 1 : 0;

  for(int t=0; t<3; t++){
    local_Nt[t] = local_N[(t + shift) % 3];
    local_Nt_start[t] = local_N_start[(t + shift) % 3];
  }

  PX(init_input_complex_3d)(N, local_Nt, local_Nt_start,
      data);
}

void PNX(init_f)(
    INT local_M,
    C *data
    )
{
  for (INT j=0; j<local_M; j++){
    R real = 100.0 * (R) rand() / RAND_MAX;
    R imag = 100.0 * (R) rand() / RAND_MAX;
    data[j] = real + imag * I;
  }
}

void PNX(init_x_3d)(
    const R *lo, const R *up, INT loc_M,
    R *x
    )
{
  R x_max[3] = {0.5,0.5,0.5};

  PNX(init_x_3d_adv)(lo, up, x_max, loc_M,
      x);
}


static void print_complex_vector(
    R *data, INT N
    )
{
  for(INT l=0; l<N; l++){
    if(l%4 == 0)
      printf("\n%4td.", l/4);
#ifdef PNFFT_PREC_LDOUBLE
    printf(" %.2Le+%.2Lei,", data[2*l], data[2*l+1]);
#else
    printf(" %.2e+%.2ei,", data[2*l], data[2*l+1]);
#endif
  }
  printf("\n");
}


void PNX(vpr_complex)(
    C *data, INT N,
    const char *name, MPI_Comm comm
    )
{
  int size, myrank;

  if(N < 1)
    return;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &myrank);
  
  fflush(stdout);
  MPI_Barrier(comm);
  for(int t=0; t<size; t++){
    if(t==myrank){
      printf("\nRank %d, %s", myrank, name);
      print_complex_vector((R*) data, N);
      fflush(stdout);
    }
    MPI_Barrier(comm);
  }
}


void PNX(vpr_real)(
    R *data, INT N,
    const char *name, MPI_Comm comm
    )
{
  int size, myrank;

  if(N < 1)
    return;
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &myrank);
  
  fflush(stdout);
  MPI_Barrier(comm);
  for(int t=0; t<size; t++){
    if(t==myrank){
      printf("\nRank %d, %s", myrank, name);
      for(INT l=0; l<N; l++){
        if(l%8 == 0)
          printf("\n%4td.", l/8);
#ifdef PNFFT_PREC_LDOUBLE
        printf(" %Le,", data[l]);
#else
        printf(" %e,", data[l]);
#endif
      }
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(comm);
  }
}


static void apr_3d(
     R *data, INT *local_N, INT *local_N_start, unsigned pnfft_flags,
     const char *name, MPI_Comm comm, const int is_complex
     )
{
  INT local_Nt[3], local_Nt_start[3];
  int shift = (pnfft_flags & PNFFT_TRANSPOSED_F_HAT) ? 1 : 0;

  for(int t=0; t<3; t++){
    local_Nt[t] = local_N[(t + shift) % 3];
    local_Nt_start[t] = local_N_start[(t + shift) % 3];
  }

  if( is_complex )
    PX(apr_complex_3d)((C*)data, local_Nt, local_Nt_start, name, comm);
  else
    PX(apr_real_3d)(data, local_Nt, local_Nt_start, name, comm);
}


void PNX(apr_complex_3d)(
     C *data, INT *local_N, INT *local_N_start, unsigned pnfft_flags,
     const char *name, MPI_Comm comm
     )
{
  apr_3d((R*)data, local_N, local_N_start, pnfft_flags, name, comm, 1);
}


void PNX(apr_real_3d)(
     R *data, INT *local_N, INT *local_N_start, unsigned pnfft_flags,
     const char *name, MPI_Comm comm
     )
{
  apr_3d(data, local_N, local_N_start, pnfft_flags, name, comm, 0);
}

