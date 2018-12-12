import numpy as np


def plotter(log, train_loss, train_dice, val_loss, val_dice, step, model, images):

    train_loss = np.around(train_loss, decimals=5)
    train_dice = np.around(train_dice, decimals=5)
    val_loss = np.around(val_loss, decimals=5)
    val_dice = np.around(val_dice, decimals=5)

    info = {'train_loss': train_loss, 'train_dice': train_dice, 'val_loss': val_loss, 'val_dice': val_dice}

    # Log scalar values (scalar summary)
    for tag, value in info.items():
        log.scalar_summary(tag, value, step+1)

    # log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        log.histo_summary(tag, value.data.cpu().numpy(), step+1)
        log.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

    # log training images (image summary)
    # info = {'images': images.view(-1, 28, 28)[:5].cpu().numpy()}
    # for tag, images in info.items():
    #     log.image_summary(tag, images, step+1)


