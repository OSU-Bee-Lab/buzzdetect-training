import os

from matplotlib import pyplot as plt


def plot_history(history, modelname, best_epoch, path_out):
    val_losses = history.history['val_loss']
    train_losses = history.history['loss']
    loss_max = max(val_losses)

    plt.plot()
    ax = plt.gca()
    ax.set_ylim([0, loss_max])
    plt.title(f'training loss curves for model {modelname}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(['training', 'validation'], loc='upper left')
    plt.annotate(f'stopped at epoch {best_epoch} with val_loss: {round(val_losses[best_epoch], 3)}', (0, 0.05))
    plt.vlines(x=best_epoch, ymin=0, ymax=loss_max)

    plt.savefig(path_out)
    plt.close()
