import numpy as np
from os import listdir
from matplotlib import pyplot as plt


def apply_spectrum(data, pca, numinput=256, power=1.0):
    colored = data.dot(np.diag(np.power(pca.sValues[:numinput], power)))
    return colored/colored.std()


def get_params_and_errors(net, toy, nunits=256, folder='.',
                          filestart='toy', ds=1.0):
    filelist = listdir(folder)
    goodfiles = []
    firing_rates = []
    gains = []
    errors = []
    modfit = []
    peaks = []
    peakmodfits = []
    exceptions = []
    for file in filelist:
        dsflag = False
        if 'ds' in file:
            dsflag = file.split('ds')[1].startswith(str(ds))
        if file.endswith('.pickle') and file.startswith(filestart) and dsflag:
            file = folder+file
            try:
                net.load(file)
            except BaseException as ee:
                exceptions.append(file)
                continue
            try:
                fit = np.load(file + 'fit.npy')
            except FileNotFoundError:
                fit = net.modfits
            ok = net.nunits == nunits
            directtest = toy.test_fit(net.Q)
            ok = ok and not (directtest - fit[-1]) > 0.01 and fit[-1] != 0
            if ok:
                modfit.append(fit[-1])
                err = np.mean(net.errorhist[-1000:])
                goodfiles.append(file)
                errors.append(err)
                firing_rates.append(net.p)
                gains.append(net.gain)
                peaks.append(np.min(net.errorhist))
                peakmodfits.append(np.max(fit))
            else:
                exceptions.append(file)
    print('Errors on ', str(len(exceptions)), ' files')
    if len(goodfiles) == 0:
        if len(exceptions) == 0:
            raise FileNotFoundError('No valid files found.')
        raise BaseException(exceptions[0])
    return (goodfiles, firing_rates, gains, errors, peaks, modfit, peakmodfits)


def hp_scatter(firing_rates, gains, modfits, fig=None, ax=None):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)
    modfits = [0 if np.isnan(mf) else mf for mf in modfits]
    sc = ax.scatter(firing_rates, gains, c=modfits, cmap='viridis', s=200)
    ax.set_xlabel('Firing rate p')
    ax.set_ylabel('Gain')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([np.min(firing_rates)*0.8, np.max(firing_rates)*1.2])
    ax.set_ylim([np.min(gains)*0.8, np.max(gains)*1.2])
    fig.colorbar(sc, ax=ax)


def err_hp_scatter(firing_rates, gains, errors, fig=None, ax=None):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)
    goodfr = [firing_rates[ii] for ii in range(len(errors)) if errors[ii] < 1.0]
    goodg = [gains[ii] for ii in range(len(errors)) if errors[ii] < 1.0]
    badfr = [firing_rates[ii] for ii in range(len(errors)) if errors[ii] >= 1.0 or np.isnan(errors[ii])]
    badg = [gains[ii] for ii in range(len(errors)) if errors[ii] >= 1.0 or np.isnan(errors[ii])]
    errors = [er for er in errors if er < 1.0]
    sc = ax.scatter(goodfr, goodg, c=errors, cmap='viridis_r', s=200)
    fig.colorbar(sc, ax=ax)
    ax.set_xlabel('Firing rate p')
    ax.set_ylabel('Gain')
    ax.scatter(badfr, badg, c='r', s=50, marker='x')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([np.min(firing_rates)*0.8, np.max(firing_rates)*1.2])
    ax.set_ylim([np.min(gains)*0.8, np.max(gains)*1.2])


def Q_and_svals(Q, pca, ds=1.0, ax=None, errorbars=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    means = np.abs(Q).mean(0)
    scale = np.max(means)
    means /= scale
    if errorbars:
        stds = np.abs(Q).std(0)/scale
        qline, = ax.errorbar(np.arange(Q.shape[0]), means, yerr=stds, fmt='b.')
    else:
        qline, = ax.plot(np.arange(Q.shape[1]), means, 'b.')
    svals = np.power(pca.sValues[:Q.shape[1]], ds)
    svals /= np.max(svals)
    sline, = ax.plot(svals, 'g')
    # ax.set_title('SAILnet PC usage follows singular values')
    ax.set_xlabel('Singular value rank')
    ax.set_ylabel('Normalized value')
    ax.legend([qline, sline], ['Mean ff weight magnitude', 'Singular value'])


def alt_Q_and_svals(Q, pca, ds=1.0, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    stds = np.abs(Q).std(0)
    stds /= np.max(stds)
    ax.plot(stds, 'b.')

    svals = np.power(pca.sValues[:256], ds)
    svals /= np.max(svals)
    ax.plot(svals, 'g')


def desphere_results(net, toy, pca_obj, desphere=1.0, folder='Pickles/4oc/'):

    (goodfiles, firing_rates, gains, errors, peaks, modfit,
        peakmodfits) = get_params_and_errors(net, toy, folder=folder, ds=desphere)

    fig = plt.figure()
    fitscat = fig.add_subplot(221)
    hp_scatter(firing_rates, gains, modfit, fig=fig, ax=fitscat)
    fitscat.set_title('Model fits')
    errscat = fig.add_subplot(223)
    err_hp_scatter(firing_rates, gains, errors, fig=fig, ax=errscat)
    errscat.set_title('Mean squared error')

    ind = np.nanargmax(modfit)
    print('Parameters of the best-fitting model: ')
    print('p = ', firing_rates[ind])
    print('gain = ', gains[ind])
    print('mse = ', errors[ind])
    print('model fit = ', modfit[ind])
    winner = goodfiles[ind]
    msewinner = goodfiles[np.nanargmin(errors)]

    try:
        fittrace = net.modfits
    except:
        fittrace = np.load(winner+'fit.npy')
    fitax = fig.add_subplot(222)
    fitax.plot(fittrace)
    fitax.set_xlabel('Training batches')
    fitax.set_ylabel('Model fit')
    # fitax.set_title('Time course of best model recovery')

    net.load(winner)
    Q_and_svals(net.Q, pca_obj, ds=desphere, ax=fig.add_subplot(224))

    fig.tight_layout()

    return winner, msewinner, fig
