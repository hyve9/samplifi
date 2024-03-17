#!/bin/bash

input=$1
output=$(basename $input .wav)-16bit.wav

ffmpeg -i $input -af aresample=osf=s16:dither_method=triangular_hp $output
