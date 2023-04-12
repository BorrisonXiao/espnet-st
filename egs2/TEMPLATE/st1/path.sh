MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:/$KALDI_ROOT/tools/sph2pipe_v2.5:$PATH:
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$MAIN_ROOT/espnet2/bin:$PATH

K2_ROOT=/home/cxiao7/research/espnet/tools/k2-cxiao
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH
export PYTHONPATH=$K2_ROOT/build_debug/lib:$PYTHONPATH
# for debugging
export K2_DISABLE_CHECKS=0
export K2_SYNC_KERNELS=1
export CUDA_LAUNCH_BLOCKING=1
# To fix the weird libcuda.so.1: cannot open shared object file import error
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib32/

export PYTHONPATH=/home/cxiao7/research/icefall-cxiao:$PYTHONPATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# NOTE(kamo): Source at the last to overwrite the setting
. local/path.sh
