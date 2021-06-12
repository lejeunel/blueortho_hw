#!/usr/bin/env python3
import click
import os
from glob import glob
from defaults import default_args as args
from loader import Loader
from defaults import default_args as args
import utils as utls
from looper import Looper
import torch
from network import MySegmentationModel
import yaml
from loss import DiceLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


@click.command()
@click.option('-c', '--cuda', is_flag=True, help='Use CUDA')
@click.option('-r',
              '--run_path',
              type=str,
              multiple=True,
              required=True,
              help='Path to run(s) to load model weights and config')
@click.option('-n',
              '--names',
              type=str,
              multiple=True,
              required=True,
              help='Names of method(s) to display')
@click.option('-i',
              '--in_path',
              type=str,
              required=True,
              help='Path to input data')
@click.option('--n_im_prev',
              type=int,
              default=8,
              help='Number of images to preview')
@click.option('--prev-path',
              type=str,
              default='test.png',
              help='Path to preview image')
def main(in_path: str, run_path: list, names: list, cuda: bool, n_im_prev,
         prev_path: str):
    assert len(run_path) == len(
        names), 'give as many run paths as method names'

    dset = Loader(in_path).shuffle(seed=42)
    dls = utls.do_splits(dset, args['train_split'],
                         (1 - args['train_split']) / 2,
                         (1 - args['train_split']) / 2)
    dl = dls['test']

    dl = DataLoader(dl,
                    collate_fn=dl.dataset.collate_fn,
                    batch_size=args['batch_size'])

    device = torch.device('cuda' if cuda else 'cpu')

    scores = {k: None for k in names}
    figs = {k: None for k in names}

    fig = plt.figure(figsize=(60., 10.))
    imgrid = ImageGrid(fig,
                       int("{}11".format(2 + len(names))),
                       nrows_ncols=(1, n_im_prev + 1),
                       axes_pad=0.05)
    truthgrid = ImageGrid(fig,
                          int("{}12".format(2 + len(names))),
                          nrows_ncols=(1, n_im_prev + 1),
                          axes_pad=0.05)
    outgrids = {
        k: ImageGrid(fig,
                     int("{}1{}".format(2 + len(names), i + 3)),
                     nrows_ncols=(1, n_im_prev + 1),
                     axes_pad=0.05)
        for i, k in enumerate(names)
    }

    for i, (name, path) in enumerate(zip(names, run_path)):

        print('run_path: ', path)
        cp = sorted(glob(os.path.join(path, 'cps', '*.pth.tar')))[-1]
        cp = torch.load(cp, map_location=lambda storage, loc: storage)

        with open(os.path.join(path, 'params.yml')) as file:
            coordconv = yaml.load(file, Loader=yaml.FullLoader)['coordconv']

        model = MySegmentationModel(coord_conv=coordconv)
        model.load_state_dict(cp)

        looper = Looper(model, dl, 'test', None, DiceLoss(smooth=0.), None,
                        device, None)
        scores[name] = 1 - looper.loop(0, 1)

        if i == 0:
            # plot preview images and ground-truths
            for i, (im, truth) in enumerate(zip(looper.ims, looper.truths)):
                if i >= n_im_prev:
                    break
                ax = imgrid[i + 1].imshow(im, cmap='gray')
                axt = truthgrid[i + 1].imshow(truth)
                imgrid[i].set_ylabel('Input image')
                truthgrid[i].set_ylabel('Ground truth')

        # plot outputs
        for i, out in enumerate(looper.outs):
            if i >= n_im_prev:
                break
            axo = outgrids[name][i + 1].imshow(out)
            if i == 0:
                outgrids[name][i].set_ylabel(name)

    # remove all ticks
    outgrids_ = []
    for n in names:
        outgrids_ += outgrids[n]

    for a in list(imgrid) + list(truthgrid) + list(outgrids_):
        a.set_xticks([])
        a.set_yticks([])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)

    print('scores:', scores)
    print('saving preview image to ', prev_path)
    fig.savefig(prev_path, bbox_inches='tight')


if __name__ == '__main__':
    main()
