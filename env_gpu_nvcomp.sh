source /lustre/blue2/ranka/eklasky/caesar_venv/bin/activate
module load gcc/14.2.0 2>/dev/null || true

export CAESAR_HOME=/blue/ranka/sa.nandikanti/CAESAR/install
export ADIOS2_DIR=/lustre/blue2/ranka/eklasky/ADIOS2/install
export NVCOMP_ROOT=$HOME/local/nvcomp

export GCC14_LIB=$(/apps/compilers/gcc/14.2.0/bin/g++ -print-file-name=libstdc++.so.6)
export GCC14_LIBDIR=$(dirname $GCC14_LIB)

export PYTHONPATH=$ADIOS2_DIR/blue/ranka/eklasky/caesar_venv/lib/python3.11/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$GCC14_LIBDIR:$CAESAR_HOME/lib:$ADIOS2_DIR/lib:$NVCOMP_ROOT/lib:$LD_LIBRARY_PATH
