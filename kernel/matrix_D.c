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
#include "bessel_i0.h"
#include "bspline.h"
#include "sinc.h"

/* The factor 1/n from matrix D cancels with the factor n of the inverse Fourier coefficients. */
#define PNFFT_INV_PHI_HAT_GAUSS(k,n,b) \
  pnfft_exp( PNFFT_SQR(PNFFT_PI*(R)(k)/(n)) * (b) )

#define PNFFT_PHI_HAT_GAUSS(k,n,b) \
  pnfft_exp(-PNFFT_SQR(PNFFT_PI*(R)(k)/(n)) * (b) )

/* The factor 1/n from matrix D cancels with the factor n of the inverse Fourier coefficients. */
static inline R inv_phi_hat_gauss_t(
    INT k, INT n, R b, int m
    )
{
  R sqrtb = pnfft_sqrt(b);
  
  C w = cerf( (R)m/sqrtb + PNFFT_PI * (R)k * sqrtb / (R)n * I );
  R coeff = creal(w) * pnfft_exp( -PNFFT_SQR( PNFFT_PI * (R)k / (R)n ) * b );
//   fprintf(stderr, "%1.16f\n", coeff);
  return 1.0 / coeff;
  
//   /* Variant 2 */
//   R x = 2.0 * PNFFT_PI * (R)m * (R)k / (R)n;
//   C w = w_of_z( (R)m/sqrtb * I - PNFFT_PI * sqrtb * (R)k / (R)n ) * ( cos(x) - I * sin(x) );
//   R r = creal(w);
//   R coeff = pnfft_exp(-PNFFT_SQR( PNFFT_PI * (R)k / (R)n ) * (R)b ) - r*pnfft_exp( -(R)m*(R)m/(R)b );
//   return 1.0/coeff;
  /* non truncated Gaussian: pnfft_exp( PNFFT_SQR( PNFFT_PI * (R)k / (R)n ) * b ) */
}

static inline R phi_hat_gauss_t(
    INT k, INT n, R b, int m
    )
{
  R sqrtb = pnfft_sqrt(b);
  C w = cerf( (R)m/sqrtb + PNFFT_PI * (R)k * sqrtb / (R)n * I );
  return pnfft_exp( -PNFFT_SQR( PNFFT_PI * (R)k / (R)n ) * b ) * creal(w);
}

static inline R inv_phi_hat_kaiser(
    INT k, INT n, R b, int m
    )
{
  R d = PNFFT_SQR( b ) - PNFFT_SQR( 2.0 * PNFFT_PI * (R)k / (R)n );

  /* compact support in Fourier space */
  if(d<0)
    return 0.0;

  return (d>0) ? 1.0 / PNX(bessel_i0)(m*pnfft_sqrt(d)) : 1.0;
}

static inline R phi_hat_kaiser(
    INT k, INT n, R b, int m
    )
{
  R d = PNFFT_SQR( b ) - PNFFT_SQR( 2.0 * PNFFT_PI * (R)k / (R)n );

  if(d<0)
    return 0.0;

  return (d>0) ? PNX(bessel_i0)(m*pnfft_sqrt(d)) : 1.0;
}

static inline R inv_phi_hat_bessel_i0(
    INT k, INT n, R b, int m
    )
{
  R d = PNFFT_SQR( b ) - PNFFT_SQR( 2.0 * PNFFT_PI * (R)k / (R)n );
  R r = (d<0) ? pnfft_sqrt(-d) : pnfft_sqrt(d);

  if(d<0)
    return r / pnfft_sin(m*r);

  return (d>0) ? r / pnfft_sinh(m*r) : 1.0/m;
}

static inline R phi_hat_bessel_i0(
    INT k, INT n, R b, int m
    )
{
  R d = PNFFT_SQR( b ) - PNFFT_SQR( 2.0 * PNFFT_PI * (R)k / (R)n );
  R r = (d<0) ? pnfft_sqrt(-d) : pnfft_sqrt(d);

  if(d<0)
    return pnfft_sin(m*r) / r;

  return (d>0) ? pnfft_sinh(m*r) / r : m;
}

/* The factor 1/n from matrix D cancels with the factor n of the inverse Fourier coefficients. */
#define PNFFT_INV_PHI_HAT_BSPLINE(k,n,m) \
  pnfft_pow( PNX(sinc)((k) * PNFFT_PI / (n)), K(-2.0) * (m))

#define PNFFT_PHI_HAT_BSPLINE(k,n,m) \
  pnfft_pow( PNX(sinc)((k) * PNFFT_PI / (n)), K(2.0) * (m))

static inline R inv_phi_hat_sinc_power(
    INT k, INT n, R b, int m, R *spline_coeffs
    )
{
  R d = pnfft_fabs(k * b / n);

  /* avoid division by zero for d == m */
  return (d < m) ? 1.0 / PNX(bspline)(2*m, d + m, spline_coeffs) : 0.0;
}

static inline R phi_hat_sinc_power(
    INT k, INT n, R b, int m, R *spline_coeffs
    )
{
  R d = pnfft_fabs(k * b / n);
  return PNX(bspline)(2*m, d + m, spline_coeffs);
}



/* For oversampling factor sigma==1 avoid division by zero. */
/* The factor 1/n from matrix D is computed in matrix B (There it cancels with the factor N of the window). */
#define PNFFT_INV_PHI_HAT_SINC_POWER(k,N,n,b,m,spline_coeffs) \
  ( (PNFFT_ABS(k) >= (n) - (N)/2) ? 0.0 : 1.0 / PNX(bspline)(2 * (m), (R)(k) * (b) / ((R) n) + (R)(m), (spline_coeffs)) )

#define PNFFT_PHI_HAT_SINC_POWER(k,N,n,b,m,spline_coeffs) \
  ( (PNFFT_ABS(k) >= (n) - (N)/2) ? 0.0 : PNX(bspline)(2 * (m), (R)(k) * (b) / ((R) n) + (R)(m), (spline_coeffs)) )


static void convolution_due_to_interlacing(
    const INT *n,
    const INT *local_N, const INT *local_N_start,
    unsigned pnfft_flags, int sign,
    C *inout);
static void convolution_with_general_window(
    const C *in,
    const INT *n,
    const INT *local_N, const INT *local_N_start,
    unsigned pnfft_flags,
    const PNX(plan) window_param, int sign,
    C *out);
static void convolution_with_pre_inv_phi_hat(
    const C *in,
    const INT *local_N,
    const C *pre_inv_phi_hat,
    unsigned pnfft_flags,
    C *out);
static void precompute_inv_phi_hat_general_window(
    const INT *local_N, const INT *local_N_start,
    const PNX(plan) window_param,
    C *pre_inv_phi_hat);

/* Return the inverse window Fourier coefficients.
 * Since we use tensor product structure, only the return the factor that belongs to dimension 'dim'. */
R PNX(inv_phi_hat)(
    const PNX(plan) ths, int dim, INT k
    )
{
  if((ths->pnfft_flags & PNFFT_WINDOW_GAUSSIAN) && (ths->pnfft_flags & PNFFT_USE_FK_GAUSSIAN_T))
    return inv_phi_hat_gauss_t(k, ths->n[dim], ths->b[dim], ths->m);
  else if(ths->pnfft_flags & PNFFT_WINDOW_GAUSSIAN)
    return PNFFT_INV_PHI_HAT_GAUSS(k, ths->n[dim], ths->b[dim]);
  else if(ths->pnfft_flags & PNFFT_WINDOW_BSPLINE)
    return PNFFT_INV_PHI_HAT_BSPLINE(k, ths->n[dim], ths->m);
  else if(ths->pnfft_flags & PNFFT_WINDOW_SINC_POWER)
    return inv_phi_hat_sinc_power(k, ths->n[dim], ths->b[dim], ths->m, ths->spline_coeffs);
  else if(ths->pnfft_flags & PNFFT_WINDOW_BESSEL_I0)
    return inv_phi_hat_bessel_i0(k, ths->n[dim], ths->b[dim], ths->m);
  else
    return inv_phi_hat_kaiser(k, ths->n[dim], ths->b[dim], ths->m);
}

R PNX(phi_hat)(
    const PNX(plan) ths, int dim, INT k
    )
{
  if((ths->pnfft_flags & PNFFT_WINDOW_GAUSSIAN) && (ths->pnfft_flags & PNFFT_USE_FK_GAUSSIAN_T))
    return phi_hat_gauss_t(k, ths->n[dim], ths->b[dim], ths->m);
  else if(ths->pnfft_flags & PNFFT_WINDOW_GAUSSIAN)
    return PNFFT_PHI_HAT_GAUSS(k, ths->n[dim], ths->b[dim]);
  else if(ths->pnfft_flags & PNFFT_WINDOW_BSPLINE)
    return PNFFT_PHI_HAT_BSPLINE(k, ths->n[dim], ths->m);
  else if(ths->pnfft_flags & PNFFT_WINDOW_SINC_POWER)
    return phi_hat_sinc_power(k, ths->n[dim], ths->b[dim], ths->m, ths->spline_coeffs);
  else if(ths->pnfft_flags & PNFFT_WINDOW_BESSEL_I0)
    return phi_hat_bessel_i0(k, ths->n[dim], ths->b[dim], ths->m);
  else
    return phi_hat_kaiser(k, ths->n[dim], ths->b[dim], ths->m);
}

void PNX(trafo_D)(
    PNX(plan) ths, int interlaced
    )
{
#if PNFFT_ENABLE_DEBUG
  PNX(debug_sum_print)((R*)ths->f_hat, ths->local_N[0]*ths->local_N[1]*ths->local_N[2], 1,
      "PNFFT: Sum of Fourier coefficients before deconvolution");
#endif

  /* use precomputed window Fourier coefficients if possible */
  if(ths->pnfft_flags & PNFFT_PRE_PHI_HAT){
    convolution_with_pre_inv_phi_hat(
        ths->f_hat, ths->local_N, ths->pre_inv_phi_hat_trafo, ths->pnfft_flags,
        (C*)ths->g1);
  } else {
    convolution_with_general_window(
        ths->f_hat, ths->n, ths->local_N, ths->local_N_start, ths->pnfft_flags, ths, FFTW_FORWARD,
        (C*)ths->g1);
  }

  /* interlaced NFFT needs extra modulation to revert the shift in x */
  if(interlaced)
    convolution_due_to_interlacing(
        ths->n, ths->local_N, ths->local_N_start, ths->pnfft_flags, FFTW_FORWARD,
        (C*)ths->g1);
}


void PNX(adjoint_D)(
    PNX(plan) ths, int interlaced
    )
{
  /* use precomputed window Fourier coefficients if possible */
  if(ths->pnfft_flags & PNFFT_PRE_PHI_HAT){
    convolution_with_pre_inv_phi_hat(
        (C*)ths->g1, ths->local_N, ths->pre_inv_phi_hat_adj, ths->pnfft_flags,
        ths->f_hat);
  } else {
    convolution_with_general_window(
        (C*)ths->g1, ths->n, ths->local_N, ths->local_N_start, ths->pnfft_flags, ths, FFTW_BACKWARD,
        ths->f_hat);
  }

  /* interlaced NFFT needs extra modulation to revert the shift in x */
  if(interlaced)
    convolution_due_to_interlacing(
        ths->n, ths->local_N, ths->local_N_start, ths->pnfft_flags, FFTW_BACKWARD,
        ths->f_hat);

#if PNFFT_ENABLE_DEBUG
  PNX(debug_sum_print)((R*)ths->f_hat, ths->local_N[0]*ths->local_N[1]*ths->local_N[2], 1,
      "PNFFT^H: Sum of Fourier coefficients after deconvolution");
#endif
}

static void convolution_due_to_interlacing(
    const INT *n,
    const INT *local_N, const INT *local_N_start,
    unsigned pnfft_flags, int sign,
    C *inout
    )
{
  INT k0, k1, k2, k=0;
  R h0, h1, h2;

  if(pnfft_flags & PNFFT_TRANSPOSED_F_HAT){
    /* f_hat is transposed N1 x N2 x N0 */
    for(k1=local_N_start[1]; k1<local_N_start[1] + local_N[1]; k1++){
      h1 = (R) k1/n[1];
      for(k2=local_N_start[2]; k2<local_N_start[2] + local_N[2]; k2++){
        h2 = h1 + (R) k2/n[2];
        for(k0=local_N_start[0]; k0<local_N_start[0] + local_N[0]; k0++, k++){
          h0 = h2 + (R) k0/n[0];
          inout[k] *= pnfft_cexp(-sign * PNFFT_PI * I * h0);
        }
      }
    }
  } else {
    /* f_hat is non-transposed N0 x N1 x N2 */
    for(k0=local_N_start[0]; k0<local_N_start[0] + local_N[0]; k0++){
      h0 = (R) k0/n[0];
      for(k1=local_N_start[1]; k1<local_N_start[1] + local_N[1]; k1++){
        h1 = h0 + (R) k1/n[1];
        for(k2=local_N_start[2]; k2<local_N_start[2] + local_N[2]; k2++, k++){
          h2 = h1 + (R) k2/n[2];
          inout[k] *= pnfft_cexp(-sign * PNFFT_PI * I * h2);
        }
      }
    }
  }
}

static void convolution_with_general_window(
    const C *in,
    const INT *n,
    const INT *local_N, const INT *local_N_start,
    unsigned pnfft_flags,
    const PNX(plan) window_param, int sign,
    C *out
    )
{
  INT k0, k1, k2, k=0;
  C inv_phi_x, inv_phi_xy, inv_phi_xyz;

  if(pnfft_flags & PNFFT_TRANSPOSED_F_HAT){
    /* g_hat is transposed N1 x N2 x N0 */
    for(k1=local_N_start[1]; k1<local_N_start[1] + local_N[1]; k1++){
      inv_phi_x = PNX(inv_phi_hat)(window_param, 1, k1);
      for(k2=local_N_start[2]; k2<local_N_start[2] + local_N[2]; k2++){
        inv_phi_xy = inv_phi_x * PNX(inv_phi_hat)(window_param, 2, k2);
        for(k0=local_N_start[0]; k0<local_N_start[0] + local_N[0]; k0++, k++){
          inv_phi_xyz = inv_phi_xy * PNX(inv_phi_hat)(window_param, 0, k0);
          out[k] = in[k] * inv_phi_xyz;
        }
      }
    }
  } else {
    /* g_hat is non-transposed N0 x N1 x N2 */
    for(k0=local_N_start[0]; k0<local_N_start[0] + local_N[0]; k0++){
      inv_phi_x = PNX(inv_phi_hat)(window_param, 0, k0);
      for(k1=local_N_start[1]; k1<local_N_start[1] + local_N[1]; k1++){
        inv_phi_xy = inv_phi_x * PNX(inv_phi_hat)(window_param, 1, k1);
        for(k2=local_N_start[2]; k2<local_N_start[2] + local_N[2]; k2++, k++){
          inv_phi_xyz = inv_phi_xy * PNX(inv_phi_hat)(window_param, 2, k2);
          out[k] = in[k] * inv_phi_xyz;
        }
      }
    }
  }
}

static void convolution_with_pre_inv_phi_hat(
    const C *in,
    const INT *local_N,
    const C *pre_inv_phi_hat,
    unsigned pnfft_flags,
    C *out
    )
{
  INT k0, k1, k2, k=0;
  const C *inv_phi_hat0 = pre_inv_phi_hat;
  const C *inv_phi_hat1 = inv_phi_hat0 + local_N[0];
  const C *inv_phi_hat2 = inv_phi_hat1 + local_N[1];

  if(pnfft_flags & PNFFT_TRANSPOSED_F_HAT){
    /* g_hat is transposed N1 x N2 x N0 */
    for(k1=0; k1<local_N[1]; k1++)
      for(k2=0; k2<local_N[2]; k2++)
        for(k0=0; k0<local_N[0]; k0++, k++)
          out[k] = in[k] * inv_phi_hat0[k0] * inv_phi_hat1[k1] * inv_phi_hat2[k2];
  } else {
    /* g_hat is non-transposed N0 x N1 x N2 */
    for(k0=0; k0<local_N[0]; k0++)
      for(k1=0; k1<local_N[1]; k1++)
        for(k2=0; k2<local_N[2]; k2++, k++)
          out[k] = in[k] * inv_phi_hat0[k0] * inv_phi_hat1[k1] * inv_phi_hat2[k2];
  }
}

void PNX(precompute_inv_phi_hat_trafo)(
    PNX(plan) ths,
    C *pre_phi_hat_trafo
    )
{
  precompute_inv_phi_hat_general_window(ths->local_N, ths->local_N_start, ths,
      pre_phi_hat_trafo);
}

void PNX(precompute_inv_phi_hat_adj)(
    PNX(plan) ths,
    C *pre_phi_hat_adj
    )
{
  precompute_inv_phi_hat_general_window(ths->local_N, ths->local_N_start, ths,
      pre_phi_hat_adj);
}

static void precompute_inv_phi_hat_general_window(
    const INT *local_N, const INT *local_N_start,
    const PNX(plan) window_param,
    C *pre_inv_phi_hat
    )
{
  INT l=0;

  for(INT t=0; t<3; t++)
    for(INT k=local_N_start[t]; k<local_N_start[t] + local_N[t]; k++, l++)
      pre_inv_phi_hat[l] = PNX(inv_phi_hat)(window_param, t, k);
}

