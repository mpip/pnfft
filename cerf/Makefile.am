# Library libcerf:
#   Compute complex error functions, based on a new implementation of
#   Faddeeva's w_of_z. Also provide Dawson and Voigt functions.
# 
# File lib/Makefile.am:
#   Input for the build process using GNU autotools.
# 
# Copyright:
#   (C) 2012 Massachusetts Institute of Technology
#   (C) 2013 Forschungszentrum Jülich GmbH
# 
# Licence:
#   MIT Licence. See ./COPYING.
# 
# Authors:
#   Steven G. Johnson, Massachusetts Institute of Technology, 2012, core author
#   Joachim Wuttke, Forschungszentrum Jülich, 2013, package maintainer
# 
# Website:
#   http://apps.jcns.fz-juelich.de/libcerf

# Directory of ipnfft.h
AM_CPPFLAGS = -I$(top_srcdir)/kernel 

# Directory of pnfft.h
AM_CPPFLAGS += -I$(top_srcdir)/api 

noinst_LTLIBRARIES = libcerf.la

# include_HEADERS = cerf.h

libcerf_la_SOURCES = \
	cerf.h \
	im_w_of_x.c \
       	erfcx.c \
	w_of_z.c \
	err_fcts.c

