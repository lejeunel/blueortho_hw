#!/usr/bin/env python3
import torch
import utils as utls
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import Normalize


class Looper:
    def __init__(self,
                 model,
                 dataloader,
                 mode,
                 optimizer,
                 criterion,
                 writer,
                 device,
                 lr_sch=None):

        assert mode in ['train', 'val',
                        'test'], 'mode must be train, val, or test'

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.mode = mode
        self.criterion = criterion
        self.lr_sch = lr_sch
        self.writer = writer
        self.device = device

    def loop(self, epoch, n_epochs):

        running_loss = 0

        # this stores images and outputs for plots
        self.ims = []
        self.truths = []
        self.outs = []

        if self.mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        pbar = tqdm(total=len(self.dataloader))
        for i, sample in enumerate(self.dataloader):
            with torch.set_grad_enabled(self.mode == 'train'):

                sample = utls.batch_to_device(sample, self.device)
                out = self.model(sample['image'])
                truth = sample['mask']
                loss = self.criterion(out, sample['mask'])

                if self.mode == 'train':
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.cpu().detach().numpy().item()

                bs = self.dataloader.batch_size
                loss = running_loss / ((i + 1) * bs)
                niter = epoch * len(self.dataloader) + i

                if self.writer:
                    self.writer.add_scalar('{}/loss'.format(self.mode), loss,
                                           niter)
                pbar.set_description(
                    '[{}] ep {}/{} lr {:.1e} lss {:.3e}'.format(
                        self.mode, epoch + 1, n_epochs,
                        self.lr_sch.get_last_lr()[0] if self.lr_sch else 0.,
                        loss))
                pbar.update(1)

                if self.mode == 'train':
                    self.optimizer.zero_grad()
                else:

                    self.ims += [
                        np.moveaxis(i.cpu().detach().numpy(), 0, -1)
                        for i in sample['image']
                    ]
                    self.truths += [
                        t.cpu().detach().numpy().squeeze() for t in truth
                    ]
                    self.outs += [
                        o.sigmoid().cpu().detach().numpy().squeeze()
                        for o in out
                    ]

        pbar.close()

        if self.lr_sch:
            self.lr_sch.step()

        # return mean of loss
        return running_loss / (len(self.dataloader) * bs)

    def make_plot(self, n_im_prev=8):

        fig = plt.figure(figsize=(30., 10.))
        imgrid = ImageGrid(fig, 311, nrows_ncols=(1, n_im_prev), axes_pad=0.05)
        truthgrid = ImageGrid(fig,
                              312,
                              nrows_ncols=(1, n_im_prev),
                              cbar_location="bottom",
                              cbar_mode="each",
                              axes_pad=(0.2, 0.05))
        # share_all=True)
        outgrid = ImageGrid(fig,
                            313,
                            nrows_ncols=(1, n_im_prev),
                            cbar_location="bottom",
                            cbar_mode="each",
                            axes_pad=(0.2, 0.05))

        try:
            for i, (im, truth,
                    out) in enumerate(zip(self.ims, self.truths, self.outs)):
                if i >= n_im_prev:
                    break
                ax = imgrid[i].imshow(im, cmap='gray')

                axt = truthgrid[i].imshow(truth, norm=Normalize())
                cax = truthgrid.cbar_axes[i]
                cb = cax.colorbar(axt)
                cb.set_ticks([truth.min(), truth.max()])
                cb.set_ticklabels(
                    ['{:.1f}'.format(x)
                     for x in [truth.min(), truth.max()]])

                axo = outgrid[i].imshow(out, norm=Normalize())
                cax = outgrid.cbar_axes[i]
                cb = cax.colorbar(axo)
                cb.set_ticks([out.min(), out.max()])
                cb.set_ticklabels(
                    ['{:.1f}'.format(x)
                     for x in [out.min(), out.max()]])

                imgrid[i].set_xticks([])
                imgrid[i].set_yticks([])
                truthgrid[i].set_xticks([])
                truthgrid[i].set_yticks([])
                outgrid[i].set_xticks([])
                outgrid[i].set_yticks([])

        except AttributeError as e:
            raise Exception(
                "No data is stored here, Run loop function first!") from e

        return fig
