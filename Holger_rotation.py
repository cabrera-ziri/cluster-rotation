import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from numba import njit

@njit(parallel=True)
def getrotprofile(pa, ra, dec, rv, erv, dis, NPPMXL):
    nstar = NPPMXL

    # Sort stars according to central distance
    index = np.argsort(dis)
    dis = dis[index]
    rv = rv[index]
    erv = erv[index]
    pa = np.deg2rad(pa[index])
    ra = ra[index]
    dec = dec[index]

    nbinpa = 360 #it would be good to have 360 bins, but 200 is also okay if 360 is too much
    lnlambdamax = -1E10

    # iterating over theta, sigma, vr to find the best fitting solution
    for i in range(nbinpa):
        theta = (i + 0.5) * 2.0 * np.pi / nbinpa
        for i2 in range(nbinpa):
            sigma = 10.0 * (i2 + 0.5) / nbinpa
            for i3 in range(nbinpa):
                vr = 10.0 * (i3 + 0.5) / nbinpa
                lnlambda = 0.0
                for j in range(nstar):
                    lnlambda += -0.5 * (np.log(sigma ** 2 + erv[j] ** 2) + (rv[j] - vr * np.sin(pa[j] - theta)) ** 2 / (sigma ** 2 + erv[j] ** 2))

                if lnlambda > lnlambdamax:

                    lnlambdamax = lnlambda
                    thetamax = theta
                    sigmamax = sigma
                    vrmax = vr


    # and now for the errors
    # first vrot
    for i in range(nbinpa):
        vr = vrmax + 4.0 * (i + 0.5) / nbinpa
        lnlambda = 0.0
        for j in range(nstar):
            lnlambda += -0.5 * (np.log(sigmamax ** 2 + erv[j] ** 2) + (rv[j] - vr * np.sin(pa[j] - thetamax)) ** 2 / (sigmamax ** 2 + erv[j] ** 2))
        if lnlambda < lnlambdamax - 0.5:
            break
    vrup = vr

    for i in range(nbinpa):
        vr = vrmax - 4.0 * (i + 0.5) / nbinpa
        lnlambda = 0.0
        for j in range(nstar):
            lnlambda += -0.5 * (np.log(sigmamax ** 2 + erv[j] ** 2) + (rv[j] - vr * np.sin(pa[j] - thetamax)) ** 2 / (sigmamax ** 2 + erv[j] ** 2))
        if lnlambda < lnlambdamax - 0.5:
            break
    vrdown = vr

    evr = (vrup - vrdown) / 2.0

    # sigma
    for i in range(nbinpa):
        sigma = sigmamax + 4.0 * (i + 0.5) / nbinpa
        lnlambda = 0.0
        for j in range(nstar):
            lnlambda += -0.5 * (np.log(sigma ** 2 + erv[j] ** 2) + (rv[j] - vrmax * np.sin(pa[j] - thetamax)) ** 2 / (sigma ** 2 + erv[j] ** 2))
        if lnlambda < lnlambdamax - 0.5:
            break
    sigmaup = sigma

    for i in range(nbinpa):
        sigma = sigmamax - 4.0 * (i + 0.5) / nbinpa
        lnlambda = 0.0
        for j in range(nstar):
            lnlambda += -0.5 * (np.log(sigma ** 2 + erv[j] ** 2) + (rv[j] - vrmax * np.sin(pa[j] - thetamax)) ** 2 / (sigma ** 2 + erv[j] ** 2))
        if lnlambda < lnlambdamax - 0.5:
            break
    sigmadown = sigma

    esigma = (sigmaup - sigmadown) / 2.0

    # theta
    for i in range(nbinpa):
        theta = thetamax + 1.0 * (i + 0.5) / nbinpa
        lnlambda = 0.0
        for j in range(nstar):
            lnlambda += -0.5 * (np.log(sigmamax ** 2 + erv[j] ** 2) + (rv[j] - vrmax * np.sin(pa[j] - theta)) ** 2 / (sigmamax ** 2 + erv[j] ** 2))
        if lnlambda < lnlambdamax - 0.5:
            break
    thetaup = theta

    for i in range(nbinpa):
        theta = thetamax - 1.0 * (i + 0.5) / nbinpa
        lnlambda = 0.0
        for j in range(nstar):
            lnlambda += -0.5 * (np.log(sigmamax ** 2 + erv[j] ** 2) + (rv[j] - vrmax * np.sin(pa[j] - theta)) ** 2 / (sigmamax ** 2 + erv[j] ** 2))
        if lnlambda < lnlambdamax - 0.5:
            break
    thetadown = theta

    etheta = (thetaup - thetadown) / 2.0

    # print(f"Best fitting values: sigma = {sigmamax:.2f} +- {esigma:.2f}   vr = {vrmax:.2f} +- {evr:.2f} theta = {np.degrees(thetamax):.1f} +- {np.degrees(etheta):.1f} degrees")

    return sigmamax, esigma, vrmax, evr, np.degrees(thetamax),np.degrees(etheta)



dforbs = pd.read_csv('orbital_params.csv')
dforbs["Clusternum"] = dforbs["Clusternum"].astype(str)
dfrvparams = pd.read_excel('RV_parameters.xlsx')
dfrvparams["Cluster"] = dfrvparams["Cluster"].astype(str)

# for s in range(0,len(dfrvparams)):
for s in range(1):
    s = 25
    num = str(int(dfrvparams.Cluster[s]))
    print("Processing cluster NGC {x}".format(x=num))
    Vc = dforbs['RV'][dforbs['Clusternum'] == num].values[0]
    Vcerr = dforbs['ERV'][dforbs['Clusternum'] == num].values[0]

    fontsize = 14

    # Read in all stars
    dft = pd.read_csv('{x}.csv'.format(x=num))
    # Read in all stars matched with photometry
    dfp = pd.read_csv('{x}_MP.csv'.format(x=num))
    dfp1 = dfp[dfp['pop']=='P1']
    dfp2 = dfp[dfp['pop']=='P2']

    fig,ax = plt.subplots(ncols=1,nrows=3,figsize=(10,10),sharex=True,sharey=True)
    axis = 0
    for df,label,color in zip([dft.copy(),dfp1.copy(),dfp2.copy()],['All stars', 'P1', 'P2'],['g','b','r']):

        # Holger's code won't work with NaN values
        df.dropna(subset='RV',inplace=True)
        df.sort_values(by=['r'], inplace=True)

        ''' Position angle '''
        b = []
        df.reset_index(inplace=True, drop=True)
        for i in range(len(df)):
            b.append((math.degrees(math.atan2(df['ra_c'][i], df['dec_c'][i])) + 360) % 360)
        df['theta'] = b

        ra = np.array(df['ra'])
        dec = np.array(df['dec'])
        dis = np.array(df['dist_cntr'])
        rv = np.array(df['RV']-Vc)
        erv = np.array(df['ERV'])
        pa = np.array(df['theta'])
        NPPMXL = len(ra)

        sigmamax, esigma, At, At_err, theta0, theta0_err = getrotprofile(pa, ra, dec, rv, erv, dis, NPPMXL)

        # You can ignore this if you don't want to plot everything
        ax[axis].errorbar(pa, rv, yerr=erv,marker='.',color='k',capsize=3,alpha=0.2,linestyle='None')
        xtemp = np.linspace(0,360,100)
        ytemp = abs(At) * np.sin(np.deg2rad(xtemp - theta0))
        ax[axis].plot(xtemp,ytemp,color=color,zorder=2)
        ax[axis].fill_between(x=xtemp, y1=ytemp - At_err, y2=ytemp + At_err, color=color, alpha=0.3,linewidth=0,zorder=2)
        anchored_text = AnchoredText(
            r'$\Delta \rm{A}$' + '$={x}\pm {z}$ [km/s]'.format(x=np.round(At, 2), z=np.round(At_err, 2)),
            loc='upper right', prop=dict(fontsize=fontsize - 3))
        ax[axis].add_artist(anchored_text)
        ax[axis].tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax[axis].plot(theta0, 0, color='white',marker='*',ms=15,zorder=12)
        ax[axis].plot(theta0, 0, color='black',marker='*',ms=22,zorder=11)
        # This is all for making the rotation angle errorbars loop around 0<->360 if necessary
        if (theta0 - theta0_err) < 0:
            ax[axis].scatter(360 - abs(theta0 - theta0_err), 0, c='white', marker='|', s=500, lw=3, zorder=3)
            ax[axis].scatter(theta0 + theta0_err, 0, c='white', marker='|', s=500, lw=3, zorder=3)
            ax[axis].hlines(xmin=360 - abs(theta0 - theta0_err), xmax=360, y=0, color='white', lw=3, zorder=3)
            ax[axis].hlines(y=0, xmin=0, xmax=theta0 + theta0_err, color='white', lw=3, zorder=3)
            # shadow underneath
            ax[axis].scatter(360 - abs(theta0 - theta0_err), 0, c='k', marker='|', s=700, lw=6, zorder=2)
            ax[axis].scatter(theta0 + theta0_err, 0, c='k', marker='|', s=700, lw=6, zorder=2)
            ax[axis].hlines(xmin=360 - abs(theta0 - theta0_err), xmax=360, y=0, color='k', lw=6, zorder=2)
            ax[axis].hlines(y=0, xmin=0, xmax=theta0 + theta0_err, color='k', lw=6, zorder=2)
        elif (theta0 + theta0_err) > 360:
            ax[axis].scatter(abs(360 - abs(theta0 + theta0_err)), 0, c='white', marker='|', s=500, lw=3, zorder=3)
            ax[axis].scatter(theta0 - theta0_err, 0, c='white', marker='|', s=500, lw=3, zorder=3)
            ax[axis].hlines(xmin=theta0 - theta0_err, xmax=360, y=0, color='white', lw=3, zorder=3)
            ax[axis].hlines(y=0, xmin=0, xmax=abs(360 - abs(theta0 + theta0_err)), color='white', lw=3, zorder=3)
            # shadow underneath
            ax[axis].scatter(abs(360 - abs(theta0 + theta0_err)), 0, c='k', marker='|', s=700, lw=6, zorder=2)
            ax[axis].scatter(theta0 - theta0_err, 0, c='k', marker='|', s=700, lw=6, zorder=2)
            ax[axis].hlines(xmin=theta0 - theta0_err, xmax=360, y=0, color='k', lw=6, zorder=2)
            ax[axis].hlines(y=0, xmin=0, xmax=abs(360 - abs(theta0 + theta0_err)), color='k', lw=6, zorder=2)
        else:
            ax[axis].scatter(theta0 + theta0_err, 0, c='white', marker='|', s=500, lw=3, zorder=3)
            ax[axis].scatter(theta0 - theta0_err, 0, c='white', marker='|', s=500, lw=3, zorder=3)
            ax[axis].hlines(y=0, xmin=theta0 - theta0_err, xmax=theta0 + theta0_err, color='white', lw=3, zorder=3)
            # shadow underneath
            ax[axis].scatter(theta0 + theta0_err, 0, c='k', marker='|', s=700, lw=6, zorder=2)
            ax[axis].scatter(theta0 - theta0_err, 0, c='k', marker='|', s=700, lw=6, zorder=2)
            ax[axis].hlines(y=0, xmin=theta0 - theta0_err, xmax=theta0 + theta0_err, color='k', lw=6, zorder=2)

        ax[axis].set_ylabel(r'$\rm{\Delta V_{LOS}}$', fontsize=fontsize)
        ax[axis].set_xlim([0,360])
        axis += 1
    plt.xlabel(r'Position angle $\theta_0$',fontsize=fontsize)
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.savefig('figures/{x}.png'.format(x=num),dpi=200,bbox_inches='tight')
    plt.show()