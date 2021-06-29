import os
from os.path import join

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from torch.nn import Sequential

from utils import batch_to_plottable_image


def main_loop(args, model, train_, val_, dl_train, dl_val, optimizer, lr_scheduler, criterion, writer, checkpoint_folder):
    """
    This is the main training loop. I left in this function the two main loops for the epochs and batches respectively.
    Since we implementing different training techniques, the actual body of the training is located in methods/ package,
    where we can more clearly organize the different training strategies, without having one single training body,
    riddled with flags and checks. The reference to the function we use for each method can be found in the
    'train_method' structure above.
    """
    best_acc = 0
    # Training Epoch Loop
    for epoch in range(args.epochs):

        # Epoch progress bar
        tq = tqdm.tqdm(total=len(dl_train) * args.train_batch_size)
        tq.set_description(f'epoch {epoch + 1}')

        train_running_loss = []
        train_running_corr = 0

        model.train()
        # Training Batch loop
        dl_train_size = len(dl_train)
        for i, (img, lbl) in enumerate(dl_train):
            '''
            All the important training stuff can be found following the train_ function. train_ function contains the
            value train_method corresponding to the args.method key. 
            '''
            # plt.imshow(batch_to_plottable_image(img))
            # plt.show()
            # exit()

            # with torch.set_grad_enabled(True):
            loss, corr = train_(model, img, lbl, optimizer, criterion)
            # Update progress bar
            tq.update(args.train_batch_size)
            tq.set_postfix({'train_loss': f'{loss:6f}', 'train_acc': f'{(corr/args.train_batch_size):.6f}' })
            train_running_loss.append(loss)
            train_running_corr += corr

            writer.add_scalar('training_loss', train_running_loss[-1], epoch * args.train_batch_size + i)
            writer.add_scalar('training_acc', corr/args.train_batch_size, epoch * args.train_batch_size + i)

        # Update learning rate
        lr_scheduler.step()

        # Close batch progress bar
        writer.add_scalar('epoch_loss', np.mean(train_running_loss), epoch)
        writer.add_scalar('epoch_acc', train_running_corr/(dl_train_size * args.train_batch_size), epoch)

        val_running_loss = []
        val_running_corr = 0

        model.eval()
        # Validation Batch Loop
        dl_val_size = len(dl_val)
        for i, (img, lbl) in enumerate(dl_val):
            with torch.set_grad_enabled(False):
                loss, corr = val_(model, img, lbl, criterion)

            val_running_loss.append(loss)
            val_running_corr += corr

        # Write on tensorboard
        writer.add_scalar('val_loss', np.mean(val_running_loss), epoch)
        writer.add_scalar('val_acc',  corr/args.val_batch_size, epoch)

        # Update progress bar
        tq.set_postfix({'train_loss': f'{np.mean(train_running_loss):.6f}',
                        'train_acc': f'{train_running_corr/(args.train_batch_size*dl_train_size):.6f}',
                        'val_loss': f'{np.mean(val_running_loss):.6f}',
                        'val_acc': f'{val_running_corr/(args.val_batch_size*dl_val_size):.6f}'})
        tq.close()

        # Save checkpoints
        new_acc = val_running_corr/(args.val_batch_size*dl_val_size)
        if new_acc > best_acc:
            best_acc = new_acc
            torch.save(Sequential(*list(model.backbone.children())[:-1]).state_dict(),
                       join(args.checkpoint_path, checkpoint_folder, f'best.pth'))

        torch.save(model.state_dict(), join(args.checkpoint_path, checkpoint_folder, f'latest.pth'))

# Training finished
