#!/usr/bin/env python
import os
import gpustat
import argparse
from loguru import logger
import time

def gpu_stat_wait_until_free(free_memory:int=3000)->int:
    gpustats = gpustat.GPUStatCollection.new_query()
    g_json = gpustats.jsonify()
    for i, g in enumerate(g_json["gpus"]):
        if(g["memory.total"]-g["memory.used"] > free_memory):
            os.environ['CUDA_VISIBLE_DEVICES'] = f'{i}'
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
        default=11500,
        help='amount of free ram needed')
    argparser.add_argument(
        '-p', '--project',
        type=str,
        default="x2",
        help='project postfix displayed in wandb')
    argparser.add_argument(
        '-d', '--data',
        type=str,
        default="data.yaml",
        help='path to data yaml description')
    argparser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=10,
        help='batchsize')
    args = argparser.parse_args()

    logger.info(vars(args))
    index = -1
    while index < 0:
        time.sleep(1)
        index = gpu_stat_wait_until_free(args.free_memory)

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{index}'
    # os.system(f"export CUDA_VISIBLE_DEVICES={index}")
    time.sleep(0.1)

    for data in [("/home/abaur/data/yolov5_dvs_c80_seq/data.yaml", "models/dvs_F.yaml", 5), ("/home/abaur/data/yolov5_rgb_c50_seq/data.yaml", "models/yolov5_dvs.yaml", 1)]:
      for method in ["l1_unstructured", "ln_structured"]:
        for pa in [0.1, 0.5, 0.75, 0.9, 0.95]:
          command = f"python train.py --img 600 --batch 20 --epochs 50 --data {data[0]} --weights '' " \
                    f"--cache --cfg {data[1]} --wandblog True --project dvs_img_finally_pruning --name {method} --time_seq_len {data[2]} --prune t --prune_amount {pa}"

          if method == "l1_unstructured":
              logger.info("running:")
              logger.info(command)
              os.system(command)
          elif method== "ln_structured":
            for pn in [1,2]:
                command += f" --prune_norm_number {pn}"

                logger.info("running:")
                logger.info(command)
                os.system(command)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info('\ndone.')

