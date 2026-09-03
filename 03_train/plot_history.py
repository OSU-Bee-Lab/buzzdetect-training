import math
import warnings

from matplotlib import pyplot as plt


def plot_history(history, modelname, best_epoch, path_out):
    # the shipped model trains to a fixed epoch count with no held-out fold to
    # monitor, so it has no val_loss to plot
    val_losses = history.history.get('val_loss')
    train_losses = history.history['loss']

    # Fit the y-axis to the data, both series together. Anchoring at 0 wastes
    # ~90% of the height here -- label_smoothing=0.2 keeps BCE well above 0, so
    # a [0, max] axis squashes every curve into a flat sliver at the top and
    # clips whichever series max() didn't see. Pad 5% of the spread each way.
    series = [x for x in list(train_losses) + list(val_losses or []) if math.isfinite(x)]
    if not series:
        # A diverged run (NaN/Inf loss) has no plottable axis. Skip the figure
        # rather than crash the whole CV loop on set_ylim; the caller still
        # raises on the NaN itself.
        warnings.warn(f'plot_history: {modelname} has no finite loss values; skipping loss_curves.svg')
        return
    lo, hi = min(series), max(series)
    pad = (hi - lo) * 0.05 or 0.01

    fig, ax = plt.subplots()
    ax.set_ylim([lo - pad, hi + pad])
    ax.set_title(f'training loss curves for model {modelname}')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.plot(train_losses, label='training')
    if val_losses is not None:
        ax.plot(val_losses, label='validation')
        note = f'stopped at epoch {best_epoch} with val_loss: {round(val_losses[best_epoch], 3)}'
    else:
        note = f'trained {best_epoch + 1} fixed epochs (consensus across folds)'
    ax.axvline(x=best_epoch, color='k', linewidth=1)
    ax.annotate(note, (0.02, 0.02), xycoords='axes fraction')
    ax.legend(loc='upper left')

    fig.savefig(path_out)
    plt.close(fig)


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
        # fit to the data, not [0, max] -- see plot_history
        lo, hi = min(val_losses), max(val_losses)
        pad = (hi - lo) * 0.05 or 0.01
        ax_loss.set_ylim([lo - pad, hi + pad])
        ax_loss.legend(loc='lower right')

    # where EarlyStopping actually restored to, for comparison against the peaks
    ax.axvline(x=best_epoch, color='k', linewidth=1)
    ax.annotate(f'restored epoch {best_epoch + 1}', (best_epoch, 0.02))
    ax.legend(loc='upper left')

    fig.savefig(path_out)
    plt.close(fig)
    return True
