#!/bin/bash
for method in super tent heaviside tanh circ heavi_erfc; do
  for cfg in dvs_F.yaml dvs_C1.yaml dvs_C3.yaml dvs_FCC.yaml dvs_FCBNCSPC.yaml; do
    for t in f t; do
      for vrs in f t; do
            command="python train.py --img 416 --batch 13 --epochs 4000 --data data.yaml --weights '' --cache --cfg  models/$cfg --wandblog True --project dvs_$1 --name $method --tauskip $t --v_resetskip $vrs --spiking_method $method"
            echo $command
            $command

      done
    done
  done
done

