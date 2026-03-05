import argparse
import os
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--rstpreid_anno_path', type=str, default='')
    parser.add_argument('--val_dataset', type=str, default='test')
    args_cli = parser.parse_args()

    args = load_train_configs(args_cli.config_file)
    args.training = False
    args.val_dataset = args_cli.val_dataset

    if args_cli.rstpreid_anno_path:
        args.rstpreid_anno_path = args_cli.rstpreid_anno_path

    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=args_cli.ckpt)
    model = model.cuda()

    do_inference(model, test_img_loader, test_txt_loader)


if __name__ == '__main__':
    main()
