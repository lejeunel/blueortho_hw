#!/usr/bin/env python3

import train
import click
import os
from defaults import default_args as args


@click.command()
@click.option('-c', '--cuda', is_flag=True, help='Use CUDA')
@click.option('-o',
              '--out_path',
              type=str,
              required=True,
              help='Path to output')
@click.option('-i',
              '--in_path',
              type=str,
              required=True,
              help='Path to output')
def main(in_path: str, out_path: str, cuda: bool):

    args['cuda'] = cuda

    args['coordconv'] = False
    args['bce'] = True
    out_path = os.path.join(out_path, 'bce')
    if not os.path.exists(out_path):
        print('training baseline')
        train.train(in_path=in_path, out_path=out_path, **args)
    else:
        print('dir ', out_path, 'found. Skipping.')

    args['coordconv'] = False
    args['bce'] = False
    out_path = os.path.join(out_path, 'dice')
    if not os.path.exists(out_path):
        print('training model with Dice loss')
        train.train(in_path=in_path,
                    out_path=os.path.join(out_path, 'dice'),
                    **args)
    else:
        print('dir ', out_path, 'found. Skipping.')

    args['coordconv'] = True
    args['bce'] = False
    out_path = os.path.join(out_path, 'dice_cc')
    if not os.path.exists(out_path):
        print('training model with Dice loss and coordconv')
        train.train(in_path=in_path,
                    out_path=os.path.join(out_path, 'dice_cc'),
                    **args)
    else:
        print('dir ', out_path, 'found. Skipping.')


if __name__ == '__main__':
    main()
