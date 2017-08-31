#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

mkdir out

python -B ./pose_seq.py --verbose True --config "msrc12_resnext_learn_comb_unc_do" --clean_start False --only_val True
mkdir out/00
mv /tmp/*.txt out/00

for i in `seq 1 4`
do
python -B ./pose_seq.py --verbose True --config "msrc12_resnext_learn_comb_unc_do_s$i" --clean_start False --only_val True
mkdir `printf "out/%02d" $i`
mv /tmp/*.txt `printf "out/%02d" $i`
done
