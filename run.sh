#!/bin/bash
#DSUB -n ml
#DSUB -A root.project.P24Z10200N0983_tmp
#DSUB -q root.default
#DSUB -R cpu=64;mem=30000;gpu=2
#DSUB -N 1
#DSUB -oo /home/share/huadjyin/home/sunhaotong/01_genomicsLLM-sukui/logs/ml.%J.out
#DSUB -eo /home/share/huadjyin/home/sunhaotong/01_genomicsLLM-sukui/logs/ml.%J.err

export PATH=/home/HPCBase/compiler/gcc-9.3.0/bin:/home/HPCBase/mpi/openmpi-4.1.2-gcc-9.3.0/bin:$PATH
export LD_LIBRARY_PATH=/home/HPCBase/compiler/gcc-9.3.0/lib64:/home/HPCBase/mpi/openmpi-4.1.2-gcc-9.3.0/lib:$LD_LIBRARY_PATH


source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
module load libs/nccl/2.18.3_cuda11
# module add compilers/gcc/9.3.0
module add compilers/gcc/14.2.0
module add libs/cudnn/8.4.0.27_cuda11.x
module add compilers/cuda/11.8.0
module add libs/openblas/0.3.25_gcc9.3.0

source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

conda activate evo

umask 000

# python /home/share/huadjyin/home/sunhaotong/01_genomicsLLM-sukui/iProbiotics/run_retrain_rawseq.py \
#     --test_folder_path=/home/share/huadjyin/home/s_kangqiang/01_data/Raw_Sequence/test/ \
#     --data_save_path=/home/share/huadjyin/home/s_kangqiang/01_data/iProbiotics/norm/ \

# python /home/share/huadjyin/home/sunhaotong/01_genomicsLLM-sukui/iProbiotics/run_retrain_1ksplit.py \
#     --test_folder_path=/home/share/huadjyin/home/s_kangqiang/01_data/Task/sample_task/test/ \
#     --data_save_path=/home/share/huadjyin/home/s_kangqiang/01_data/iProbiotics/run_1ksplit_test/ \

torchrun --nproc_per_node=2 --master_port=22400 /home/share/huadjyin/home/sunhaotong/02_SPP_FMRESAC/main.py\
    model_name=NT_50M\
    input_path=/home/share/huadjyin/home/s_kangqiang/01_data/Raw_Sequence/LR13/\
    output_path=/home/share/huadjyin/home/s_kangqiang/01_data/LR13/\