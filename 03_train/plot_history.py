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


def plot_sens_history(history, modelname, best_epoch, fprs, key, path_out):
    """sens@FPR per epoch, with val_loss on a twin axis.

    Kept out of loss_curves.svg deliberately: the question these answer is
    whether val_loss's minimum lands anywhere near sens@FPR's peak, and that
    reads better on its own axes than as two more lines on a loss plot. Returns
    False and writes nothing when no fold-level target was ever reachable.
    """
    curves = {f: history.history.get(key(f)) for f in fprs}
    curves = {f: c for f, c in curves.items() if c and not all(v != v for v in c)}
    if not curves:
        return False

    fig, ax = plt.subplots()
    for fpr, curve in curves.items():
        ax.plot(curve, label=f'sens@fpr{fpr:.1%}')
    ax.set_ylim([0, 1])
    ax.set_xlabel('epoch')
    ax.set_ylabel('sensitivity')
    ax.set_title(f'held-out sens@FPR per epoch for model {modelname}')

    val_losses = history.history.get('val_loss')
    if val_losses is not None:
        ax_loss = ax.twinx()
        ax_loss.plot(val_losses, color='0.6', linestyle=':', label='val_loss')
        ax_loss.set_ylabel('val_loss')
        ax_loss.set_ylim([0, max(val_losses)])
        ax_loss.legend(loc='lower right')

    # where EarlyStopping actually restored to, for comparison against the peaks
    ax.axvline(x=best_epoch, color='k', linewidth=1)
    ax.annotate(f'restored epoch {best_epoch + 1}', (best_epoch, 0.02))
    ax.legend(loc='upper left')

    fig.savefig(path_out)
    plt.close(fig)
    return True
