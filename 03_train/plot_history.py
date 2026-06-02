import os

from matplotlib import pyplot as plt


def plot_history(history, modelname, best_epoch, path_out):
    val_losses = history.history['val_loss']
    epoch_loss = val_losses[best_epoch]
    loss_max = max(val_losses)

    plt.plot()
    ax = plt.gca()
    ax.set_ylim([0, loss_max])
    plt.title(f'training loss curves for model {modelname}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.annotate(f'stopped at epoch {best_epoch} with val_loss: {round(epoch_loss, 3)}', (0, 0.05))
    plt.vlines(x=best_epoch, ymin=0, ymax=loss_max)

    plt.savefig(path_out)
    plt.close()
