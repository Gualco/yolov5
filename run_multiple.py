#!/usr/bin/env python
import os
import gpustat
import argparse
from loguru import logger
import time

def gpu_stat_wait(free_memory:int=3000):
    gpustats = gpustat.GPUStatCollection.new_query()
    g_json = gpustats.jsonify()
    for i, g in enumerate(g_json["gpus"]):
        if(g["memory.total"]-g["memory.used"] > free_memory):
            return i
        else:
            # print("fucking fully loaded GPU")
            pass

    return -1

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-m', '--free_memory',
        type=int,
        default=3000,
        help='amount of free ram needed')
    argparser.add_argument(
        '-p', '--project',
        type=str,
        default="x2",
        help='project postfix displayed in wandb')
    args = argparser.parse_args()

    logger.info(vars(args))
    index = gpu_stat_wait(args.free_memory)
    while index < 0:
        time.sleep(1)
        index = gpu_stat_wait(args.free_memory)

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{index}'
    # os.system(f"export CUDA_VISIBLE_DEVICES={index}")
    time.sleep(0.1)

    for method in ["super", "tent", "heaviside", "tanh circ", "heavi_erfc"][:1]:
      for cfg in ["dvs_FCBNCSPC.yaml", "dvs_FCBNCSPC_x2.yaml", "dvs_AllConv.yaml", "dvs_AllConv_x2.yaml", "dvs_F.yaml", "dvs_F_x2.yaml", "dvs_C1_x2.yaml",
                  "dvs_C1_x2.yaml", "dvs_C3.yaml", "dvs_FCC.yaml"][:2]:
        for t in ["f", "t"][:1]:
          for vrs in ["f", "t"][:1]:
            command="python train.py --img 416 --batch 13 --epochs 4000 --data data.yaml --weights '' " \
                        f"--cache --cfg  models/{cfg} --wandblog True --project dvs_{args.project} --name {method} --tauskip {t} --v_resetskip {vrs} --spiking_method {method}"
            os.system(command)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info('\ndone.')

