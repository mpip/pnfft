#!/bin/sh -e

myprefix=$HOME/local
INSTDIR=$myprefix/pfft
FFTWDIR=$myprefix/fftw
TMP="${PWD}/tmp-pfft"
LOGFILE="${TMP}/build.log"

# bash check if directory exists
if [ -d $TMP ]; then
        echo "Directory $TMP already exists. Delete it? (y/n)"
	read answer
	if [ ${answer} = "y" ]; then
		rm -rf $TMP
	else
		echo "Program aborted."
		exit 1
	fi
fi

mkdir $TMP && cd $TMP


git clone --branch=master git://github.com/mpip/pfft.git pfft

cd pfft

./bootstrap.sh && \
./configure --prefix=$INSTDIR \
  CPPFLAGS="-I$FFTWDIR/include" \
  LDFLAGS="-L$FFTWDIR/lib" \
  FC=mpif90 CC=mpicc MPICC=mpicc MPIFC=mpif90 2>&1 | tee $LOGFILE && \
make -j 4 2>&1 | tee -a $LOGFILE && \
make install 2>&1 | tee -a $LOGFILE
