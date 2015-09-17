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

static void spread_f_c2c_pre_psi(
    C f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grid);
static void spread_f_c2c_pre_full_psi(
    C f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grid);
static void spread_f_r2r_pre_psi(
    R f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT ostride,
    R *grid);
static void spread_f_r2r_pre_full_psi(
    R f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT ostride,
    R *grid);

static void spread_grad_f_c2c_pre_psi(
    const C *grad_f, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grid);
static void spread_grad_f_c2c_pre_full_psi(
    const C *grad_f, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grid);
static void spread_grad_f_r2r_pre_psi(
    const R *grad_f, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *grid);
static void spread_grad_f_r2r_pre_full_psi(
    const R *grad_f, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *grid);

static void assign_f_c2c_pre_psi(
    const C *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *fv);
static void assign_f_c2c_pre_full_psi(
    const C *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *fv);
static void assign_f_r2r_pre_psi(
    const R *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride,
    R *fv);
static void assign_f_r2r_pre_full_psi(
    const R *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, int istride,
    R *fv);

static void assign_grad_f_c2c_pre_psi(
    const C *grid, R *pre_psi, R *pre_dpsi, 
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grad_f);
static void assign_grad_f_c2c_pre_full_psi(
    const C *grid, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grad_f);
static void assign_grad_f_r2r_pre_psi(
    const R *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *grad_f);
static void assign_grad_f_r2r_pre_full_psi(
    const R *grid, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *grad_f);

static void assign_hessian_f_c2c_pre_psi(
    const C *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *hessian_f);
static void assign_hessian_f_c2c_pre_full_psi(
    const C *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *hessian_f);
static void assign_hessian_f_r2r_pre_psi(
    const R *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *hessian_f);
static void assign_hessian_f_r2r_pre_full_psi(
    const R *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *hessian_f);

static void assign_f_and_grad_f_c2c_pre_psi(
    const C *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *fv, C *grad_f);
static void assign_f_and_grad_f_c2c_pre_full_psi(
    const C *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *fv, C *grad_f);
static void assign_f_and_grad_f_r2r_pre_psi(
    const R *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *fv, R *grad_f);
static void assign_f_and_grad_f_r2r_pre_full_psi(
    const R *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *fv, R *grad_f);





void PNX(spread_f_c2c)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    C f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff,
    int use_interlacing, int interlaced,
    C *grid
    )
{
  R* plan_pre_psi = (interlaced) ? nodes->pre_psi_il : nodes->pre_psi;

  if( ~nodes->precompute_flags & PNFFT_PRE_PSI )
    spread_f_c2c_pre_psi(
        f, pre_psi, m0, grid_size, cutoff, use_interlacing,
        grid);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    spread_f_c2c_pre_full_psi(
        f, plan_pre_psi + ind*PNFFT_POW3(cutoff), m0, grid_size, cutoff, use_interlacing, 
        grid);
  else
    spread_f_c2c_pre_psi(
        f, plan_pre_psi + ind*3*cutoff, m0, grid_size, cutoff, use_interlacing, 
        grid);
}

void PNX(spread_f_r2r)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    R f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, INT ostride,
    int use_interlacing, int interlaced,
    R *grid
    )
{
  R* plan_pre_psi = (interlaced) ? nodes->pre_psi_il : nodes->pre_psi;

  if( ~nodes->precompute_flags & PNFFT_PRE_PSI )
    spread_f_r2r_pre_psi(
        f, pre_psi, m0, grid_size, cutoff, ostride, use_interlacing,
        grid);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    spread_f_r2r_pre_full_psi(
        f, plan_pre_psi + ind*PNFFT_POW3(cutoff), m0, grid_size, cutoff, ostride, use_interlacing,
        grid);
  else
    spread_f_r2r_pre_psi(
        f, plan_pre_psi + ind*3*cutoff, m0, grid_size, cutoff, ostride, use_interlacing,
        grid);
}

void PNX(spread_grad_f_c2c)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const C *grad_f, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff,
    int use_interlacing, int interlaced,
    C *grid
    )
{
  R* plan_pre_psi  = (interlaced) ? nodes->pre_psi_il  : nodes->pre_psi;
  R* plan_pre_dpsi = (interlaced) ? nodes->pre_dpsi_il : nodes->pre_dpsi;

  if( ~nodes->precompute_flags & PNFFT_PRE_GRAD_PSI )
    spread_grad_f_c2c_pre_psi(
        grad_f, pre_psi, pre_dpsi, m0, grid_size, cutoff, use_interlacing, 
        grid);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    spread_grad_f_c2c_pre_full_psi(
        grad_f, plan_pre_dpsi + ind*PNFFT_POW3(cutoff),
        m0, grid_size, cutoff, use_interlacing, 
        grid);
  else
    spread_grad_f_c2c_pre_psi(
        grad_f, plan_pre_psi + ind*3*cutoff, plan_pre_dpsi + ind*3*cutoff, 
        m0, grid_size, cutoff, use_interlacing, 
        grid);
}

void PNX(spread_grad_f_r2r)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const R *grad_f, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff,
    INT istride, INT ostride,
    int use_interlacing, int interlaced,
    R *grid
    )
{
  R* plan_pre_psi  = (interlaced) ? nodes->pre_psi_il  : nodes->pre_psi;
  R* plan_pre_dpsi = (interlaced) ? nodes->pre_dpsi_il : nodes->pre_dpsi;

  if( ~nodes->precompute_flags & PNFFT_PRE_GRAD_PSI )
    spread_grad_f_r2r_pre_psi(
        grad_f, pre_psi, pre_dpsi,
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        grid);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    spread_grad_f_r2r_pre_full_psi(
        grad_f, plan_pre_dpsi + ind*PNFFT_POW3(cutoff),
        m0, grid_size, cutoff, use_interlacing, istride, ostride, 
        grid);
  else
    spread_grad_f_r2r_pre_psi(
        grad_f, plan_pre_psi + ind*3*cutoff, plan_pre_dpsi + ind*3*cutoff, 
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        grid);
}

void PNX(assign_f_c2c)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const C *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff,
    int use_interlacing, int interlaced,
    C *f
    )
{
  R* plan_pre_psi = (interlaced) ? nodes->pre_psi_il  : nodes->pre_psi;

  if( ~nodes->precompute_flags & PNFFT_PRE_PSI )
    assign_f_c2c_pre_psi(
        grid, pre_psi, m0, grid_size, cutoff, use_interlacing,
        f);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    assign_f_c2c_pre_full_psi(
        grid, plan_pre_psi + ind*PNFFT_POW3(cutoff), m0, grid_size, cutoff, use_interlacing,
        f);
  else
    assign_f_c2c_pre_psi(
        grid, plan_pre_psi + ind*3*cutoff, m0, grid_size, cutoff, use_interlacing,
        f);
}

void PNX(assign_f_r2r)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const R *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, INT istride,
    int use_interlacing, int interlaced,
    R *f
    )
{ 
  R* plan_pre_psi = (interlaced) ? nodes->pre_psi_il  : nodes->pre_psi;

  if( ~nodes->precompute_flags & PNFFT_PRE_PSI )
    assign_f_r2r_pre_psi(
        grid, pre_psi, m0, grid_size, cutoff, use_interlacing, istride,
        f);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    assign_f_r2r_pre_full_psi(
        grid, plan_pre_psi + ind*PNFFT_POW3(cutoff), m0, grid_size, cutoff, use_interlacing, istride,
        f);
  else
    assign_f_r2r_pre_psi(
        grid, plan_pre_psi + ind*3*cutoff, m0, grid_size, cutoff, use_interlacing, istride,
        f);
}

void PNX(assign_grad_f_c2c)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const C *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff,
    int use_interlacing, int interlaced,
    C *grad_f
    )
{
  R* plan_pre_psi  = (interlaced) ? nodes->pre_psi_il  : nodes->pre_psi;
  R* plan_pre_dpsi = (interlaced) ? nodes->pre_dpsi_il : nodes->pre_dpsi;

  if( ~nodes->precompute_flags & PNFFT_PRE_GRAD_PSI )
    assign_grad_f_c2c_pre_psi(
        grid, pre_psi, pre_dpsi,
        m0, grid_size, cutoff, use_interlacing,
        grad_f);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    assign_grad_f_c2c_pre_full_psi(
        grid, plan_pre_dpsi + 3*ind*PNFFT_POW3(cutoff),
        m0, grid_size, cutoff, use_interlacing,
        grad_f);
  else
    assign_grad_f_c2c_pre_psi(
        grid, plan_pre_psi + ind*3*cutoff, plan_pre_dpsi + ind*3*cutoff,
        m0, grid_size, cutoff, use_interlacing,
        grad_f);
}

void PNX(assign_grad_f_r2r)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const R *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff,
    INT istride, INT ostride,
    int use_interlacing, int interlaced,
    R *grad_f
    )
{
  R* plan_pre_psi  = (interlaced) ? nodes->pre_psi_il  : nodes->pre_psi;
  R* plan_pre_dpsi = (interlaced) ? nodes->pre_dpsi_il : nodes->pre_dpsi;

  if( ~nodes->precompute_flags & PNFFT_PRE_GRAD_PSI )
    assign_grad_f_r2r_pre_psi(
        grid, pre_psi, pre_dpsi,
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        grad_f);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    assign_grad_f_r2r_pre_full_psi(
        grid, plan_pre_dpsi + 3*ind*PNFFT_POW3(cutoff),
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        grad_f);
  else
    assign_grad_f_r2r_pre_psi(
        grid, plan_pre_psi + ind*3*cutoff, plan_pre_dpsi + ind*3*cutoff,
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        grad_f);
}

void PNX(assign_hessian_f_c2c)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const C *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff,
    int use_interlacing, int interlaced,
    C *hessian_f
    )
{
  R* plan_pre_psi   = (interlaced) ? nodes->pre_psi_il   : nodes->pre_psi;
  R* plan_pre_dpsi  = (interlaced) ? nodes->pre_dpsi_il  : nodes->pre_dpsi;
  R* plan_pre_ddpsi = (interlaced) ? nodes->pre_ddpsi_il : nodes->pre_ddpsi;

  if( ~nodes->precompute_flags & PNFFT_PRE_HESSIAN_PSI )
    assign_hessian_f_c2c_pre_psi(
        grid, pre_psi, pre_dpsi, pre_ddpsi,
        m0, grid_size, cutoff, use_interlacing,
        hessian_f);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    assign_hessian_f_c2c_pre_full_psi(
        grid,
        plan_pre_psi + ind*PNFFT_POW3(cutoff),
        plan_pre_dpsi + 3*ind*PNFFT_POW3(cutoff),
        plan_pre_ddpsi + 3*ind*PNFFT_POW3(cutoff),
        m0, grid_size, cutoff, use_interlacing,
        hessian_f);
  else
    assign_hessian_f_c2c_pre_psi(
        grid, 
        plan_pre_psi + ind*3*cutoff,
        plan_pre_dpsi + ind*3*cutoff,
        plan_pre_ddpsi + ind*3*cutoff,
        m0, grid_size, cutoff, use_interlacing,
        hessian_f);
}

void PNX(assign_hessian_f_r2r)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const R *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff,
    INT istride, INT ostride,
    int use_interlacing, int interlaced,
    R *hessian_f
    )
{
  R* plan_pre_psi   = (interlaced) ? nodes->pre_psi_il   : nodes->pre_psi;
  R* plan_pre_dpsi  = (interlaced) ? nodes->pre_dpsi_il  : nodes->pre_dpsi;
  R* plan_pre_ddpsi = (interlaced) ? nodes->pre_ddpsi_il : nodes->pre_ddpsi;

  if( ~nodes->precompute_flags & PNFFT_PRE_HESSIAN_PSI )
    assign_hessian_f_r2r_pre_psi(
        grid, pre_psi, pre_dpsi, pre_ddpsi,
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        hessian_f);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    assign_hessian_f_r2r_pre_full_psi(
        grid,
        plan_pre_psi + ind*PNFFT_POW3(cutoff),
        plan_pre_dpsi + 3*ind*PNFFT_POW3(cutoff),
        plan_pre_ddpsi + 3*ind*PNFFT_POW3(cutoff),
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        hessian_f);
  else
    assign_hessian_f_r2r_pre_psi(
        grid, 
        plan_pre_psi + ind*3*cutoff, 
        plan_pre_dpsi + ind*3*cutoff,
        plan_pre_ddpsi + ind*3*cutoff,
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        hessian_f);
}




void PNX(assign_f_and_grad_f_c2c)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const C *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff,
    int use_interlacing, int interlaced,
    C *f, C *grad_f
    )
{ 
  R* plan_pre_psi  = (interlaced) ? nodes->pre_psi_il  : nodes->pre_psi;
  R* plan_pre_dpsi = (interlaced) ? nodes->pre_dpsi_il : nodes->pre_dpsi;

  if( ~nodes->precompute_flags & PNFFT_PRE_GRAD_PSI )
    assign_f_and_grad_f_c2c_pre_psi(
        grid, pre_psi, pre_dpsi,
        m0, grid_size, cutoff, use_interlacing,
        f, grad_f);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    assign_f_and_grad_f_c2c_pre_full_psi(
        grid, plan_pre_psi + ind*PNFFT_POW3(cutoff), plan_pre_dpsi + 3*ind*PNFFT_POW3(cutoff),
        m0, grid_size, cutoff, use_interlacing,
        f, grad_f);
  else
    assign_f_and_grad_f_c2c_pre_psi(
        grid, plan_pre_psi + ind*3*cutoff, plan_pre_dpsi + ind*3*cutoff,
        m0, grid_size, cutoff, use_interlacing,
        f, grad_f);
}

void PNX(assign_f_and_grad_f_r2r)(
    PNX(plan) ths, PNX(nodes) nodes, INT ind,
    const R *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff,
    INT istride, INT ostride,
    int use_interlacing, int interlaced,
    R *f, R *grad_f
    )
{ 
  R* plan_pre_psi  = (interlaced) ? nodes->pre_psi_il  : nodes->pre_psi;
  R* plan_pre_dpsi = (interlaced) ? nodes->pre_dpsi_il : nodes->pre_dpsi;

  if( ~nodes->precompute_flags & PNFFT_PRE_GRAD_PSI )
    assign_f_and_grad_f_r2r_pre_psi(
        grid, pre_psi, pre_dpsi,
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        f, grad_f);
  else if (nodes->precompute_flags & PNFFT_PRE_FULL)
    assign_f_and_grad_f_r2r_pre_full_psi(
        grid, plan_pre_psi + ind*PNFFT_POW3(cutoff), plan_pre_dpsi + 3*ind*PNFFT_POW3(cutoff),
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        f, grad_f);
  else
    assign_f_and_grad_f_r2r_pre_psi(
        grid, plan_pre_psi + ind*3*cutoff, plan_pre_dpsi + ind*3*cutoff,
        m0, grid_size, cutoff, use_interlacing, istride, ostride,
        f, grad_f);
}








static void spread_f_c2c_pre_psi(
    C f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grid
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff];

  if(use_interlacing) f *= 0.5;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      R psi_xy = pre_psi_x[l0] * pre_psi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++ ){
        grid[m2] += psi_xy * pre_psi_z[l2] * f;
      }
    }
  }
}

static void spread_f_c2c_pre_full_psi(
    C f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grid
    )
{ 
  INT m1, m2, l0, l1, l2, m=0;

  if(use_interlacing) f *= 0.5;
  
  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2])
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2])
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++, m++ )
        grid[m2] += pre_psi[m] * f;
}

static void spread_f_r2r_pre_psi(
    R f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT ostride,
    R *grid
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff];

  if(use_interlacing) f *= 0.5;
  
  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*ostride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*ostride){
      R psi_xy = pre_psi_x[l0] * pre_psi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=ostride ){
        grid[m2] += psi_xy * pre_psi_z[l2] * f;
      }
    }
  }
}

static void spread_f_r2r_pre_full_psi(
    R f, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT ostride,
    R *grid
    )
{ 
  INT m1, m2, l0, l1, l2, m=0;

  if(use_interlacing) f *= 0.5;
  
  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*ostride)
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*ostride)
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=ostride, m++ )
        grid[m2] += pre_psi[m] * f;
}



static void spread_grad_f_c2c_pre_psi(
    const C *grad_f, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grid
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff], *pre_dpsi_x = &pre_dpsi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff], *pre_dpsi_y = &pre_dpsi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff], *pre_dpsi_z = &pre_dpsi[2*cutoff];
  C g0 = grad_f[0], g1 = grad_f[1], g2 = grad_f[2];

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }
  
  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      R psi_xy  = pre_psi_x[l0]  * pre_psi_y[l1];
      R psi_dxy = pre_dpsi_x[l0] * pre_psi_y[l1]; 
      R psi_xdy = pre_psi_x[l0]  * pre_dpsi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++ ){
        grid[m2] -= psi_dxy * pre_psi_z[l2]  * g0;
        grid[m2] -= psi_xdy * pre_psi_z[l2]  * g1;
        grid[m2] -= psi_xy  * pre_dpsi_z[l2] * g2;
      }
    }
  }
}

static void spread_grad_f_c2c_pre_full_psi(
    const C *grad_f, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grid
    )
{ 
  INT m1, m2, l0, l1, l2, dm=0;
  C g0 = grad_f[0], g1 = grad_f[1], g2 = grad_f[2];

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }
  
  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++, dm+=3 ){
        grid[m2] -= pre_dpsi[dm+0] * g0;
        grid[m2] -= pre_dpsi[dm+1] * g1;
        grid[m2] -= pre_dpsi[dm+2] * g2;
      }
    }
  }
}

static void spread_grad_f_r2r_pre_psi(
    const R *grad_f, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *grid
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff], *pre_dpsi_x = &pre_dpsi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff], *pre_dpsi_y = &pre_dpsi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff], *pre_dpsi_z = &pre_dpsi[2*cutoff];
  R g0 = grad_f[0*istride], g1 = grad_f[1*istride], g2 = grad_f[2*istride];

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }
  
  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*ostride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*ostride){
      R psi_xy  = pre_psi_x[l0]  * pre_psi_y[l1];
      R psi_dxy = pre_dpsi_x[l0] * pre_psi_y[l1]; 
      R psi_xdy = pre_psi_x[l0]  * pre_dpsi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=ostride ){
        grid[m2] -= psi_dxy * pre_psi_z[l2]  * g0;
        grid[m2] -= psi_xdy * pre_psi_z[l2]  * g1;
        grid[m2] -= psi_xy  * pre_dpsi_z[l2] * g2;
      }
    }
  }
}

static void spread_grad_f_r2r_pre_full_psi(
    const R *grad_f, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *grid
    )
{ 
  INT m1, m2, l0, l1, l2, m=0;
  R g0 = grad_f[0*istride], g1 = grad_f[1*istride], g2 = grad_f[2*istride];

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }
  
  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*ostride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*ostride){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=ostride, m++ ){
        grid[m2] -= pre_dpsi[m] * g0;
        grid[m2] -= pre_dpsi[m] * g1;
        grid[m2] -= pre_dpsi[m] * g2;
      }
    }
  }
}


static void assign_f_c2c_pre_psi(
    const C *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *fv
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff];
  C f=0;
  
  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      R psi_xy = pre_psi_x[l0] * pre_psi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++ ){
        f += psi_xy * pre_psi_z[l2] * grid[m2];
      }
    }
  }

  if(use_interlacing) f *= 0.5;

  *fv += f;
}

static void assign_f_c2c_pre_full_psi(
    const C *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *fv
    )
{ 
  INT m1, m2, l0, l1, l2, m=0;
  C f=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++, m++ ){
        f += pre_psi[m] * grid[m2];
      }
    }
  }

  if(use_interlacing) f *= 0.5;

  *fv += f;
}

static void assign_f_r2r_pre_psi(
    const R *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride,
    R *fv
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff];
  R f=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*istride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*istride){
      R psi_xy = pre_psi_x[l0] * pre_psi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=istride ){
        f += psi_xy * pre_psi_z[l2] * grid[m2];
      }
    }
  }

  if(use_interlacing) f *= 0.5;

  *fv += f;
}

static void assign_f_r2r_pre_full_psi(
    const R *grid, R *pre_psi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, int istride,
    R *fv
    )
{ 
  INT m1, m2, l0, l1, l2, m=0;
  R f=0;
  
  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*istride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*istride){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=istride, m++ ){
        f += pre_psi[m] * grid[m2];
      }
    }
  }

  if(use_interlacing) f *= 0.5;

  *fv += f;
}

static void assign_grad_f_c2c_pre_psi(
    const C *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grad_f
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff], *pre_dpsi_x = &pre_dpsi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff], *pre_dpsi_y = &pre_dpsi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff], *pre_dpsi_z = &pre_dpsi[2*cutoff];
  C g0=0, g1=0, g2=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      R psi_xy  = pre_psi_x[l0]  * pre_psi_y[l1];
      R psi_dxy = pre_dpsi_x[l0] * pre_psi_y[l1]; 
      R psi_xdy = pre_psi_x[l0]  * pre_dpsi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++ ){
        g0 += psi_dxy * pre_psi_z[l2]  * grid[m2];
        g1 += psi_xdy * pre_psi_z[l2]  * grid[m2];
        g2 += psi_xy  * pre_dpsi_z[l2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }

  grad_f[0] += g0; grad_f[1] += g1; grad_f[2] += g2;
}

static void assign_grad_f_c2c_pre_full_psi(
    const C *grid, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *grad_f
    )
{ 
  INT m1, m2, l0, l1, l2, dm=0;
  C g0=0, g1=0, g2=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++, dm+=3 ){
        g0 += pre_dpsi[dm+0] * grid[m2];
        g1 += pre_dpsi[dm+1] * grid[m2];
        g2 += pre_dpsi[dm+2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }

  grad_f[0] += g0; grad_f[1] += g1; grad_f[2] += g2;
}

static void assign_grad_f_r2r_pre_psi(
    const R *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *grad_f
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff], *pre_dpsi_x = &pre_dpsi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff], *pre_dpsi_y = &pre_dpsi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff], *pre_dpsi_z = &pre_dpsi[2*cutoff];
  R g0=0, g1=0, g2=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*istride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*istride){
      R psi_xy  = pre_psi_x[l0]  * pre_psi_y[l1];
      R psi_dxy = pre_dpsi_x[l0] * pre_psi_y[l1]; 
      R psi_xdy = pre_psi_x[l0]  * pre_dpsi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=istride ){
        g0 += psi_dxy * pre_psi_z[l2]  * grid[m2];
        g1 += psi_xdy * pre_psi_z[l2]  * grid[m2];
        g2 += psi_xy  * pre_dpsi_z[l2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }

  grad_f[0*ostride] += g0; grad_f[1*ostride] += g1; grad_f[2*ostride] += g2;
}

static void assign_grad_f_r2r_pre_full_psi(
    const R *grid, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *grad_f
    )
{ 
  INT m1, m2, l0, l1, l2, dm=0;
  R g0=0, g1=0, g2=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*istride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*istride){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=istride, dm+=3 ){
        g0 += pre_dpsi[dm+0] * grid[m2];
        g1 += pre_dpsi[dm+1] * grid[m2];
        g2 += pre_dpsi[dm+2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }

  grad_f[0*ostride] += g0; grad_f[1*ostride] += g1; grad_f[2*ostride] += g2;
}



static void assign_hessian_f_c2c_pre_psi(
    const C *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *hessian_f
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff], *pre_dpsi_x = &pre_dpsi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff], *pre_dpsi_y = &pre_dpsi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff], *pre_dpsi_z = &pre_dpsi[2*cutoff];
  R *pre_ddpsi_x =  &pre_ddpsi[0*cutoff];
  R *pre_ddpsi_y =  &pre_ddpsi[1*cutoff];
  R *pre_ddpsi_z =  &pre_ddpsi[2*cutoff];
  C g0=0, g1=0, g2=0, g3=0, g4=0, g5=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      R psi_xy   = pre_psi_x[l0]   * pre_psi_y[l1];
      R psi_dxy  = pre_dpsi_x[l0]  * pre_psi_y[l1]; 
      R psi_xdy  = pre_psi_x[l0]   * pre_dpsi_y[l1];
      R psi_dxdy = pre_dpsi_x[l0]  * pre_dpsi_y[l1];
      R psi_ddxy = pre_ddpsi_x[l0] * pre_psi_y[l1];
      R psi_xddy = pre_psi_x[l0]   * pre_ddpsi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++ ){
        g0 += psi_ddxy * pre_psi_z[l2]   * grid[m2];
        g1 += psi_dxdy * pre_psi_z[l2]   * grid[m2];
        g2 += psi_dxy  * pre_dpsi_z[l2]  * grid[m2];
        g3 += psi_xddy * pre_psi_z[l2]   * grid[m2];
        g4 += psi_xdy  * pre_dpsi_z[l2]  * grid[m2];
        g5 += psi_xy   * pre_ddpsi_z[l2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
    g3 *= 0.5; g4 *= 0.5; g5 *= 0.5;
  }

  hessian_f[0] += g0; hessian_f[1] += g1; hessian_f[2] += g2;
  hessian_f[3] += g3; hessian_f[4] += g4; hessian_f[5] += g5;
}

static void assign_hessian_f_c2c_pre_full_psi(
    const C *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *hessian_f
    )
{ 
  INT m1, m2, l0, l1, l2, ddm=0;
  C g0=0, g1=0, g2=0, g3=0, g4=0, g5=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++, ddm+=6 ){
        g0 += pre_ddpsi[ddm+0] * grid[m2];
        g1 += pre_ddpsi[ddm+1] * grid[m2];
        g2 += pre_ddpsi[ddm+2] * grid[m2];
        g3 += pre_ddpsi[ddm+3] * grid[m2];
        g4 += pre_ddpsi[ddm+4] * grid[m2];
        g5 += pre_ddpsi[ddm+5] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
    g3 *= 0.5; g4 *= 0.5; g5 *= 0.5;
  }

  hessian_f[0] += g0; hessian_f[1] += g1; hessian_f[2] += g2;
  hessian_f[3] += g3; hessian_f[4] += g4; hessian_f[5] += g5;
}

static void assign_hessian_f_r2r_pre_psi(
    const R *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *hessian_f
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff], *pre_dpsi_x = &pre_dpsi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff], *pre_dpsi_y = &pre_dpsi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff], *pre_dpsi_z = &pre_dpsi[2*cutoff];
  R *pre_ddpsi_x =  &pre_ddpsi[0*cutoff];
  R *pre_ddpsi_y =  &pre_ddpsi[1*cutoff];
  R *pre_ddpsi_z =  &pre_ddpsi[2*cutoff];
  R g0=0, g1=0, g2=0, g3=0, g4=0, g5=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*istride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*istride){
      R psi_xy   = pre_psi_x[l0]   * pre_psi_y[l1];
      R psi_dxy  = pre_dpsi_x[l0]  * pre_psi_y[l1]; 
      R psi_xdy  = pre_psi_x[l0]   * pre_dpsi_y[l1];
      R psi_dxdy = pre_dpsi_x[l0]  * pre_dpsi_y[l1];
      R psi_ddxy = pre_ddpsi_x[l0] * pre_psi_y[l1];
      R psi_xddy = pre_psi_x[l0]   * pre_ddpsi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=istride ){
        g0 += psi_ddxy * pre_psi_z[l2]   * grid[m2];
        g1 += psi_dxdy * pre_psi_z[l2]   * grid[m2];
        g2 += psi_dxy  * pre_dpsi_z[l2]  * grid[m2];
        g3 += psi_xddy * pre_psi_z[l2]   * grid[m2];
        g4 += psi_xdy  * pre_dpsi_z[l2]  * grid[m2];
        g5 += psi_xy   * pre_ddpsi_z[l2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
    g3 *= 0.5; g4 *= 0.5; g5 *= 0.5;
  }

  hessian_f[0*ostride] += g0; hessian_f[1*ostride] += g1; hessian_f[2*ostride] += g2;
  hessian_f[3*ostride] += g3; hessian_f[4*ostride] += g4; hessian_f[5*ostride] += g5;
}

static void assign_hessian_f_r2r_pre_full_psi(
    const R *grid, R *pre_psi, R *pre_dpsi, R *pre_ddpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *hessian_f
    )
{ 
  INT m1, m2, l0, l1, l2, ddm=0;
  R g0=0, g1=0, g2=0, g3=0, g4=0, g5=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*istride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*istride){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=istride, ddm+=6 ){
        g0 += pre_ddpsi[ddm+0] * grid[m2];
        g1 += pre_ddpsi[ddm+1] * grid[m2];
        g2 += pre_ddpsi[ddm+2] * grid[m2];
        g3 += pre_ddpsi[ddm+3] * grid[m2];
        g4 += pre_ddpsi[ddm+4] * grid[m2];
        g5 += pre_ddpsi[ddm+5] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
    g3 *= 0.5; g4 *= 0.5; g5 *= 0.5;
  }

  hessian_f[0*ostride] += g0; hessian_f[1*ostride] += g1; hessian_f[2*ostride] += g2;
  hessian_f[3*ostride] += g3; hessian_f[4*ostride] += g4; hessian_f[5*ostride] += g5;
}





static void assign_f_and_grad_f_c2c_pre_psi(
    const C *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *fv, C *grad_f
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff], *pre_dpsi_x = &pre_dpsi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff], *pre_dpsi_y = &pre_dpsi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff], *pre_dpsi_z = &pre_dpsi[2*cutoff];
  C f=0, g0=0, g1=0, g2=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      R psi_xy  = pre_psi_x[l0] * pre_psi_y[l1];
      R psi_dxy = pre_dpsi_x[l0] * pre_psi_y[l1]; 
      R psi_xdy = pre_psi_x[l0] * pre_dpsi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++ ){
        f  += psi_xy  * pre_psi_z[l2]  * grid[m2];
        g0 += psi_dxy * pre_psi_z[l2]  * grid[m2];
        g1 += psi_xdy * pre_psi_z[l2]  * grid[m2];
        g2 += psi_xy  * pre_dpsi_z[l2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    f *= 0.5;
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }

  *fv += f;
  grad_f[0] += g0; grad_f[1] += g1; grad_f[2] += g2;
}

static void assign_f_and_grad_f_c2c_pre_full_psi(
    const C *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing,
    C *fv, C *grad_f
    )
{ 
  INT m1, m2, l0, l1, l2, m=0, dm=0;
  C f=0, g0=0, g1=0, g2=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2++, m++, dm+=3 ){
        f  += pre_psi[m]  * grid[m2];
        g0 += pre_dpsi[dm+0] * grid[m2];
        g1 += pre_dpsi[dm+1] * grid[m2];
        g2 += pre_dpsi[dm+2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    f *= 0.5;
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }

  *fv += f;
  grad_f[0] += g0; grad_f[1] += g1; grad_f[2] += g2;
}

static void assign_f_and_grad_f_r2r_pre_psi(
    const R *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *fv, R *grad_f
    )
{ 
  INT m1, m2, l0, l1, l2;
  R *pre_psi_x = &pre_psi[0*cutoff], *pre_dpsi_x = &pre_dpsi[0*cutoff];
  R *pre_psi_y = &pre_psi[1*cutoff], *pre_dpsi_y = &pre_dpsi[1*cutoff];
  R *pre_psi_z = &pre_psi[2*cutoff], *pre_dpsi_z = &pre_dpsi[2*cutoff];
  R f=0, g0=0, g1=0, g2=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*istride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*istride){
      R psi_xy  = pre_psi_x[l0] * pre_psi_y[l1];
      R psi_dxy = pre_dpsi_x[l0] * pre_psi_y[l1]; 
      R psi_xdy = pre_psi_x[l0] * pre_dpsi_y[l1];
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=istride ){
        f  += psi_xy  * pre_psi_z[l2]  * grid[m2];
        g0 += psi_dxy * pre_psi_z[l2]  * grid[m2];
        g1 += psi_xdy * pre_psi_z[l2]  * grid[m2];
        g2 += psi_xy  * pre_dpsi_z[l2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    f *= 0.5;
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }

  *fv += f;
  grad_f[0*ostride] += g0; grad_f[1*ostride] += g1; grad_f[2*ostride] += g2;
}

static void assign_f_and_grad_f_r2r_pre_full_psi(
    const R *grid, R *pre_psi, R *pre_dpsi,
    INT m0, const INT *grid_size, int cutoff, int use_interlacing, INT istride, INT ostride,
    R *fv, R *grad_f
    )
{
  INT m1, m2, l0, l1, l2, m=0, dm=0;
  R f=0, g0=0, g1=0, g2=0;

  for(l0=0; l0<cutoff; l0++, m0 += grid_size[1]*grid_size[2]*istride){
    for(l1=0, m1=m0; l1<cutoff; l1++, m1 += grid_size[2]*istride){
      for(l2=0, m2 = m1; l2<cutoff; l2++, m2+=istride, m++, dm+=3 ){
        f  += pre_psi[m]  * grid[m2];
        g0 += pre_dpsi[dm+0] * grid[m2];
        g1 += pre_dpsi[dm+1] * grid[m2];
        g2 += pre_dpsi[dm+2] * grid[m2];
      }
    }
  }

  if(use_interlacing){
    f *= 0.5;
    g0 *= 0.5; g1 *= 0.5; g2 *= 0.5;
  }

  *fv += f;
  grad_f[0*ostride] += g0; grad_f[1*ostride] += g1; grad_f[2*ostride] += g2;
}


