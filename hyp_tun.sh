#!/bin/bash
for cfg in yolov5_dvs_Covn7.yaml yolov5_dvs_focus_S.yaml; do
    for t in f t; do
        for vrs in f t; do
            command="python train.py --img 416 --batch 4 --epochs 2000 --data data.yaml --weights '' --cache --cfg  models/$cfg --wandblog True --project dvs_img --name super --tauskip $t --v_resetskip $vrs"
            echo $command
            $command
        done
    done
done
