import os
from os.path import join

import numpy as np
import torch
import tqdm
from torch.nn import Sequential


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
        tq = tqdm.tqdm(total=len(dl_train) * args.batch_size)
        tq.set_description(f'epoch {epoch + 1}')

        train_running_loss = []
        train_running_acc = []

        # Training Batch loop
        for i, (img, lbl) in enumerate(dl_train):
            '''
            All the important training stuff can be found following the train_ function. train_ function contains the
            value train_method corresponding to the args.method key. 
            '''
            loss, acc = train_(model, img, lbl, optimizer, criterion)
            # Update progress bar
            tq.update(args.batch_size)
            tq.set_postfix({'train_loss': '%.6f' % loss, 'train_acc': '%.6f' % acc})
            train_running_loss.append(loss)
            train_running_acc.append(acc)

            writer.add_scalar('training_loss', loss, epoch * args.batch_size + i)
            writer.add_scalar('training_acc', acc, epoch * args.batch_size + i)

        # Update learning rate
        lr_scheduler.step()

        # Close batch progress bar
        writer.add_scalar('epoch_loss', np.mean(train_running_loss), epoch)
        writer.add_scalar('epoch_acc', np.mean(train_running_loss), epoch)

        val_running_loss = []
        val_running_acc = []

        # Validation Batch Loop
        for i, (img, lbl) in enumerate(dl_val):
            loss, acc = val_(model, img, lbl, criterion)
            val_running_loss.append(loss)
            val_running_acc.append(acc)

        # Write on tensorboard
        writer.add_scalar('val_loss', np.mean(val_running_loss), epoch)
        writer.add_scalar('val_acc', np.mean(val_running_acc), epoch)

        # Update progress bar
        tq.set_postfix({'train_loss': '%.6f' % np.mean(train_running_loss),
                        'train_acc': '%.6f' % np.mean(train_running_acc),
                        'val_loss': '%.6f' % np.mean(val_running_loss),
                        'val_acc': '%.6f' % np.mean(val_running_acc)})
        tq.close()

        # Save checkpoints
        new_acc = np.mean(val_running_acc)
        if new_acc > best_acc:
            best_acc = new_acc
            torch.save(Sequential(*list(model.backbone.children())[:-1]).state_dict(),
                       join(args.checkpoint_path, checkpoint_folder, f'best.pth'))

        torch.save(model.state_dict(), join(args.checkpoint_path, checkpoint_folder, f'latest.pth'))

# Training finished
