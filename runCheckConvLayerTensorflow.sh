#!/usr/bin/env bash
mmf="/home/dawna/pwv20/work/chime4/exp/DR3/dnn-ce-base-dummy-set-up/hmm0/init/work/MMF"
uttName="CHiME4-00011-011C0201-XXXPED-05_XXXSIMU_0000000_0000652.wav"
./checkConvLayerTensorflow.py ${uttName} ${mmf} 

