# Subdirectories
SUBDIRS = 

# Enable Fortran subdirectory only on request and only for double precision.
if ENABLE_FORTRAN
if DOUBLE
SUBDIRS += f03
endif
endif

# Directory of ipnfft.h
AM_CPPFLAGS = -I$(top_srcdir)/kernel 

# Directory of pnfft.h
AM_CPPFLAGS += -I$(top_srcdir)/api

# Libraries to add to all programs that are built.
LDADD = $(top_builddir)/lib@PNFFT_PREFIX@pnfft@PREC_SUFFIX@.la $(pfft_LIBS) $(fftw3_mpi_LIBS) $(fftw3_LIBS)

EXTRA_DIST = \
	plot_m_vs_error.py \
	build_checks.sh \
	run_checks.sh

# These programs are built by 'make check' and may be tested afterwards.
check_PROGRAMS =

if SINGLE
check_PROGRAMS += \
	check_trafo_vs_ndft_float
endif

if LDOUBLE
check_PROGRAMS += \
	check_trafo_vs_ndft_ldouble
endif

if DOUBLE
check_PROGRAMS += \
	simple_test \
	simple_test_c2r \
	check_trafo check_adj \
	check_trafo_2d check_trafo_transposed_2d \
	check_trafo_vs_ndft_transposed_2d \
	check_vs_pfft \
	pnfft_test \
	pnfft_test_adv \
	check_trafo_vs_ndft_c2r check_adj_vs_ndft_c2r \
	simple_test_c2r_c2c_compare_real simple_test_c2r_c2c_compare_complex \
	simple_test_c2r_c2c_compare_grad simple_test_c2r_c2c_compare_timer \
	check_charge_dipole
endif

