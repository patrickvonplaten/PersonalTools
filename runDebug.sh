#!/usr/bin/env bash
rm -r hmm0
mkdir -p hmm0
gdb --args base/HTKTools/HNTrainSGD -B -A -D -V -T 3 -C dnn-ce-orig/lib/cfgs/local.cfg -C finetune.cfg -S dnn-ce-orig/lib/flists/train.scp -N dnn-ce-orig/lib/flists/cv.scp -l LABEL -I dnn-ce-orig/lib/mlabs/traincv.mlf -H dnn-ce-orig/hmm0/init/MMF -M hmm0 dnn-ce-orig/lib/mlists/model.lst
rm -r hmm0/epoch* 
