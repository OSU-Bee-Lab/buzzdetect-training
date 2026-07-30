import os

from matplotlib import pyplot as plt


def plot_history(history, modelname, best_epoch, path_out):
    # the shipped model trains to a fixed epoch count with no held-out fold to
    # monitor, so it has no val_loss to plot
    val_losses = history.history.get('val_loss')
    train_losses = history.history['loss']
    loss_max = max(val_losses if val_losses is not None else train_losses)

    plt.plot()
    ax = plt.gca()
    ax.set_ylim([0, loss_max])
    plt.title(f'training loss curves for model {modelname}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(train_losses)
    legend = ['training']
    if val_losses is not None:
        plt.plot(val_losses)
        legend.append('validation')
        plt.annotate(f'stopped at epoch {best_epoch} with val_loss: {round(val_losses[best_epoch], 3)}', (0, 0.05))
    else:
        plt.annotate(f'trained {best_epoch + 1} fixed epochs (median across folds)', (0, 0.05))
    plt.legend(legend, loc='upper left')
    plt.vlines(x=best_epoch, ymin=0, ymax=loss_max)

    plt.savefig(path_out)
    plt.close()
