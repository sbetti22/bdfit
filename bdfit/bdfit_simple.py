'''
fitspec is a wrapper class for fitting and comparing spectra using the species package for low resolution NIR spectra.  

written by: Sarah Betti 06-2025

'''

import os
from species import SpeciesInit
from species.data.database import Database
from species.fit.fit_model import FitModel
from species.read.read_model import ReadModel
from species.plot.plot_mcmc import plot_posterior
from species.plot.plot_spectrum import plot_spectrum
from species.util.box_util import update_objectbox
from species.util.fit_util import get_residuals, multi_photometry
from species.fit.compare_spectra import CompareSpectra
from species.plot.plot_comparison import plot_empirical_spectra
from species.plot.plot_comparison import plot_model_spectra

from astropy import units as u
from astropy import constants

import corner
import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.25')

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)



def run_bdfit_mcmc(OBJ_NAME, PARALLAX, PARALLAX_err, specfile_full_name,
                  bounds,  n_live_points=2000, teff_range=(2100., 3600.), save_name = None, 
                 spec_res=2700., model='bt-settl-cifist',error=True,**kwargs):
    spec_res = float(spec_res)
    database = Database()

    database.add_object(OBJ_NAME,
                            parallax=(PARALLAX, PARALLAX_err),
                            flux_density=None,
                            spectrum={f'{OBJ_NAME}': (specfile_full_name, None, spec_res),},
                            deredden=None)

    database.add_model(model=model, teff_range=teff_range)

    fit = FitModel(object_name=OBJ_NAME,
            model=model,
            bounds=bounds,
            inc_phot=False,
            inc_spec=[OBJ_NAME],
            fit_corr=None,
            apply_weights=False, ext_model='CCM89') 
        
    fit.run_multinest(tag=OBJ_NAME,
                n_live_points=n_live_points,
                resume=False,
                output='multinest/',
                kwargs_multinest=None)


    fig = plot_posterior(tag=OBJ_NAME,
                    offset=(-0.3 , -0.3),
                    title_fmt='.2f',
                    inc_luminosity=True,
                    inc_mass=False,
                    output=f'{save_name}_mcmc_posterior')
        

    box_plotting = database.get_samples(tag=OBJ_NAME)
    samples_plotting = box_plotting.samples
    params_plotting = box_plotting.parameters
    
    teff_index = np.argwhere(np.array(params_plotting) == "teff")[0]
    radius_index = np.argwhere(np.array(params_plotting) == "radius")[0]
    R_JUP = constants.R_jup.value
    L_SUN = constants.L_sun.value
    SIGMA_SB = constants.sigma_sb.value
    lum_atm_plotting = (4.0 * np.pi * (samples_plotting[..., radius_index] * R_JUP) ** 2 * SIGMA_SB * samples_plotting[..., teff_index] ** 4.0 / L_SUN)

    ndim_plotting = len(params_plotting)

    samples_plotting = np.append(samples_plotting, np.log10(lum_atm_plotting), axis=-1)
    params_plotting.append("log_lum_atm")
    ndim_plotting += 1
    best_samples = samples_plotting.reshape((-1, ndim_plotting))
    ########

    best = database.get_median_sample(tag=OBJ_NAME)
    read_model = ReadModel(model=model, wavel_range=None)
    modelbox = read_model.get_model(model_param=best,
                                    spec_res=spec_res)
    objectbox = database.get_object(object_name=OBJ_NAME,
                                    inc_phot=False,
                                    inc_spec=[OBJ_NAME])
    objectbox = update_objectbox(objectbox=objectbox, model_param=best)
    residuals = get_residuals(tag = OBJ_NAME,
                                    parameters=best,
                                    objectbox=objectbox,
                                    inc_phot=False,
                                    inc_spec=True)
    Wave = modelbox.wavelength
    Flux = modelbox.flux
    idx = np.where(np.isfinite(Flux))
    Wave = Wave[idx].astype('double')
    Flux =Flux[idx].astype('double')
                
    final_model = np.c_[Wave, Flux]
    leg_vals = modelbox.parameters
    print('saving best fit model to: ', f'{save_name}_BTSettl_bestfit_model.txt')
    np.savetxt(f'{save_name}_BTSettl_bestfit_model.txt', final_model)
        
    _plot_bestmodel_residual(OBJ_NAME, modelbox, objectbox, residuals, save_name=save_name,  **kwargs)
    
    # ##### error ####
    # if error:
    #     Teff = modelbox.parameters["teff"]

    #     Teff_plus_offset = Teff+100.
    #     Teff_minus_offset = Teff-100.

    #     bounds_pos = bounds.copy()
    #     bounds_min = bounds.copy()

    #     if (Teff_plus_offset != bounds['teff'][0]) & (Teff_plus_offset != bounds['teff'][1]):
    #         print(f'>>> changing bound["teff"] from {bounds["teff"]} to {(Teff_plus_offset, Teff_plus_offset)}')

    #     bounds_pos['teff'] = (Teff_plus_offset, Teff_plus_offset)
    #     bounds_pos['logg'] = (modelbox.parameters["logg"]-0.5, modelbox.parameters["logg"]-0.5)
    #     fit_plus = FitModel(object_name=OBJ_NAME, model=model, bounds=bounds_pos, inc_phot=False, inc_spec=[OBJ_NAME], fit_corr=None, apply_weights=False, ext_model='CCM89')  
    #     fit_plus.run_multinest(tag=OBJ_NAME+'_plus', n_live_points=n_live_points, resume=False, output='multinest/', kwargs_multinest=None)
    #     plus_samples, plus_vals = _compute_individual_error(database, OBJ_NAME+'_plus', best)
        
       
    #     if (Teff_minus_offset != bounds['teff'][0]) & (Teff_minus_offset != bounds['teff'][1]):
    #         print(f'changing bound["teff"] from {bounds["teff"]} to {(Teff_minus_offset, Teff_minus_offset)}')
    #     bounds_min['teff'] = (Teff_minus_offset, Teff_minus_offset)
    #     bounds_min['logg'] = (modelbox.parameters["logg"]+0.5, modelbox.parameters["logg"]+0.5)
    #     fit_minus = FitModel(object_name=OBJ_NAME, model=model, bounds=bounds_min, inc_phot=False, inc_spec=[OBJ_NAME], fit_corr=None, apply_weights=False, ext_model='CCM89')  
    #     fit_minus.run_multinest(tag=OBJ_NAME+'_minus', n_live_points=n_live_points, resume=False, output='multinest/', kwargs_multinest=None)
    #     minus_samples, minus_vals = _compute_individual_error(database, OBJ_NAME+'_minus', best)
       

    #     _plot_error_corner(best_samples, plus_samples, plus_vals, minus_samples, minus_vals, Teff, save_name)

    #     name = []
    #     value = []
    #     minn = []
    #     maxx = []
    #     for key, val in plus_vals.items():
    #         if key != 'parallax':
    #             name.append(key)
    #             value.append(val[0])
    #             minn.append(minus_vals[key][1])
    #             maxx.append(val[1])
    #     name.append('chi2')
    #     value.append(residuals.chi2_red)
    #     minn.append(np.nan)
    #     maxx.append(np.nan)
    #     d = {'name': name, 'best_fit': value, 'err_lo':minn, 'err_hi':maxx}
    #     df = pd.DataFrame(d)
    #     print('save best fit parameters to: ', f'{save_name}_BTSettl_bestfit_parameters.csv')
    #     df.to_csv(f'{save_name}_BTSettl_bestfit_parameters.csv') 
    #     # display(df)    
    # else:
    #     #### save best params no error #### 
    name = []
    value = []
    best_vals = {}
    for i in np.arange(len(params_plotting)):
        name.append(params_plotting[i])
        value.append(corner.quantile(best_samples[:,i], [0.5])[0])
        best_vals[params_plotting[i]] = corner.quantile(best_samples[:,i], [0.5])[0]
    name.append('chi2')
    value.append(residuals.chi2_red)
    df = pd.DataFrame({'name': name, 'best_fit': value})
    print('save best fit parameters to: ', f'{save_name}_BTSettl_bestfit_parameters.csv')
    df.to_csv(f'{save_name}_BTSettl_bestfit_parameters.csv')   

    Teff = modelbox.parameters["teff"]
    _plot_corner(best_samples, best_vals, Teff, save_name) 
    return df        
  

def _plot_corner(best_samples, best_vals, Teff, save_name):
    print(best_vals)
    print('best fit Teff = ', Teff)

    if np.shape(best_samples)[1] == 8:
        axislabels=[r"$T_{\rm eff}$ (K)", 'log g', r"$R_p\ (R_J)$", r"parallax (mas)", 'disk teff', 'disk radius', r"$A_V$ (mag)", r"$\log L_p / L_\odot$"]
        labels=[r"$T_{\rm eff}$", 'log g', r"$R_p$", r"$\varpi$", 'Tdisk', 'Rdisk', r"$A_V$", r"$\log L_p / L_\odot$"]
        labelend = ["K", "", r"$R_J$", "mas", "K", r"$R_J$", "mag", ""]
    else:
        axislabels=[r"$T_{\rm eff}$ (K)", 'log g', r"$R_p\ (R_J)$", r"parallax (mas)",  r"$A_V$ (mag)", r"$\log L_p / L_\odot$"]
        labels=[r"$T_{\rm eff}$", 'logg', r"$R_p$", r"$\varpi$", r"$A_V$", r"$\log L_p / L_\odot$"]
        labelend = ["K", '', r"$R_J$", "mas", "mag", ""]

    minlim = []
    maxlim = []
    fig, axes = plt.subplots(nrows=len(labels), ncols=len(labels), figsize=(14,14))
    for yi in np.arange(len(labels)):
        for xi in np.arange(yi+1):
            if yi > xi:
                axes[xi, yi].axis('off')
    for i, k in enumerate(list(best_vals.keys())):
        ax = axes[i, i]
        ax.hist(best_samples[:,i], bins=20, color='k', histtype='step')
        ax.axvline(best_vals[k], color='k', linestyle='--')
        ax.set_title(labels[i] + '$=$' + str(round(best_vals[k],2)) + ' ' + labelend[i])
        
        minn = min(best_samples[:,i])
        maxx = max(best_samples[:,i])
        minlim.append(minn)
        maxlim.append(maxx)


    for yi in np.arange(len(labels)):
        for xi in np.arange(yi):
            ax = axes[yi, xi]
            corner.hist2d(best_samples[:, xi],best_samples[:, yi], ax=ax, color='k')

    for yi in np.arange(len(labels)):
        for xi in np.arange(yi+1):
            ax = axes[yi, xi] # y goes across, x goes up
            if xi == yi:
                ax.set_xlim(minlim[yi], maxlim[yi])

            else:
                ax.set_xlim(minlim[xi], maxlim[xi])
                ax.set_ylim(minlim[yi], maxlim[yi])
                
            if (xi > 0):  
                ax.set_yticklabels([])
            if yi <= len(labels)-2:
                ax.set_xticklabels([])
                
            if xi == 0:
                ax.set_ylabel(axislabels[yi])
            if yi == len(labels)-1:
                ax.set_xlabel(axislabels[xi])
            
    plt.tight_layout()
    plt.savefig(f'{save_name}_posteriorV2.pdf', dpi=150)
    plt.clf()
    # plt.show()

def _plot_error_corner(best_samples, plus_samples, plus_vals, minus_samples, minus_vals, Teff, save_name):
    print('best fit Teff = ', Teff)
    
    red = '#D81B60'
    blue = '#1E88E5'
    if np.shape(best_samples)[1] == 8:
        axislabels=[r"$T_{\rm eff}$ (K)", 'log g', r"$R_p\ (R_J)$", r"parallax (mas)", 'disk teff', 'disk radius', r"$A_V$ (mag)", r"$\log L_p / L_\odot$"]
        labels=[r"$T_{\rm eff}$", 'log g', r"$R_p$", r"$\varpi$", 'Tdisk', 'Rdisk', r"$A_V$", r"$\log L_p / L_\odot$"]
        labelend = ["K", "", r"$R_J$", "mas", "K", r"$R_J$", "mag", ""]
    else:
        axislabels=[r"$T_{\rm eff}$ (K)", 'log g', r"$R_p\ (R_J)$", r"parallax (mas)",  r"$A_V$ (mag)", r"$\log L_p / L_\odot$"]
        labels=[r"$T_{\rm eff}$", 'logg', r"$R_p$", r"$\varpi$", r"$A_V$", r"$\log L_p / L_\odot$"]
        labelend = ["K", '', r"$R_J$", "mas", "mag", ""]

    minlim = []
    maxlim = []
    fig, axes = plt.subplots(nrows=len(labels), ncols=len(labels), figsize=(14,14))
    for yi in np.arange(len(labels)):
        for xi in np.arange(yi+1):
            if yi > xi:
                axes[xi, yi].axis('off')
    for i, k in enumerate(list(plus_vals.keys())):
        ax = axes[i, i]
        ax.hist(best_samples[:,i], bins=20, color='k', histtype='step')
        ax.axvline(plus_vals[k][0], color='k', linestyle='--')
        ax.set_title(labels[i] + '$=$' + str(round(plus_vals[k][0],2)) + '$^{+' + str(round(plus_vals[k][1],2)) + '}_{-' + str(round(minus_vals[k][1],2)) + '}$ ' + labelend[i])
        if i > 1:
            if np.nanmedian(plus_samples[:,i]) < np.nanmedian(minus_samples[:,i]):
                ax.axvline(plus_vals[k][0]-plus_vals[k][1], color=red, linestyle='--')
                ax.axvline(plus_vals[k][0]+minus_vals[k][1], color=blue, linestyle='--')
            else:
                ax.axvline(plus_vals[k][0]+plus_vals[k][1], color=red, linestyle='--')
                ax.axvline(plus_vals[k][0]-minus_vals[k][1], color=blue, linestyle='--')
        
            
            ax.hist(plus_samples[:,i], bins=20, color=red, histtype='step')
            ax.hist(minus_samples[:,i], bins=20, color=blue, histtype='step')
            minn = min(min(plus_samples[:,i]), min(minus_samples[:,i]), min(best_samples[:,i]))
            maxx = max(max(plus_samples[:,i]), max(minus_samples[:,i]), max(best_samples[:,i]))
            minlim.append(minn)
            maxlim.append(maxx)
        else:
            minn = min(best_samples[:,i])
            maxx = max(best_samples[:,i])
            minlim.append(minn)
            maxlim.append(maxx)

    for yi in np.arange(len(labels)):
        for xi in np.arange(yi):
            ax = axes[yi, xi]
            corner.hist2d(best_samples[:, xi],best_samples[:, yi], ax=ax, color='k')
            if (yi > 1) & (xi>1):
                ax = axes[yi, xi]
                corner.hist2d(plus_samples[:, xi],plus_samples[:, yi], ax=ax, color=red)
                corner.hist2d(minus_samples[:, xi],minus_samples[:, yi], ax=ax, color=blue)

    for yi in np.arange(len(labels)):
        for xi in np.arange(yi+1):
            ax = axes[yi, xi] # y goes across, x goes up
            if xi == yi:
                ax.set_xlim(minlim[yi], maxlim[yi])

            else:
                ax.set_xlim(minlim[xi], maxlim[xi])
                ax.set_ylim(minlim[yi], maxlim[yi])
                
            if (xi > 0):  
                ax.set_yticklabels([])
            if yi <= len(labels)-2:
                ax.set_xticklabels([])
                
            if xi == 0:
                ax.set_ylabel(axislabels[yi])
            if yi == len(labels)-1:
                ax.set_xlabel(axislabels[xi])
            
        
    labels= [r'fit with $T_{\rm eff} - 100$ K', 'best fit', r'fit with $T_{\rm eff} + 100$ K']
    colors = [blue, 'k', red]
    plt.legend(
            handles=[
                mlines.Line2D([], [], color=colors[i], label=labels[i])
                for i in np.arange(3)
            ],
            fontsize=15, frameon=False,
            bbox_to_anchor=(1, 5), loc="upper right"
        )

    plt.tight_layout()
    plt.savefig(f'{save_name}_posterior_error.pdf', dpi=150)
    plt.clf()
    # plt.show()


def _compute_individual_error(database, tag, best):
        box = database.get_samples(tag=tag)
        samples = box.samples
        params = box.parameters
        print(params)
        ndim = len(params)
    
        teff_index = np.argwhere(np.array(box.parameters) == "teff")[0]
        radius_index = np.argwhere(np.array(box.parameters) == "radius")[0]
        R_JUP = constants.R_jup.value
        L_SUN = constants.L_sun.value
        SIGMA_SB = constants.sigma_sb.value
        lum_atm = (
            4.0
            * np.pi
            * (samples[..., radius_index] * R_JUP) ** 2
            * SIGMA_SB
            * samples[..., teff_index] ** 4.0
            / L_SUN
        )
        
        samples = box.samples[:, 0:-2]
        params = box.parameters[0:-2]
        ndim = len(params)

        samples = np.append(samples, np.log10(lum_atm), axis=-1)
        params.append("log_lum_atm")
        ndim += 1
        samples0 = samples.reshape((-1, ndim))

        
        VALS = {}
        VALS['teff'] = [best['teff'], 100]
        VALS['logg'] = [best['logg'], 0.5]
        for i in np.arange(ndim):
            q_16, q_50, q_84 = corner.quantile(samples0[:, i], [0.16, 0.5, 0.84])
            best_val = best[params[i]]
            if best_val > q_50:
                err = best_val - q_16  
            else:
                err = q_84 - best_val
            VALS[params[i]] = [best_val, err]
            print(params[i], best_val, err)
            
        samples = np.append(np.ones_like(np.log10(lum_atm)), samples, axis=-1)
        samples = np.append(np.ones_like(np.log10(lum_atm)), samples, axis=-1)
        # params.insert(0, 'teff')
        # params.insert(1, 'logg')
        ndim += 2
        samples = samples.reshape((-1, ndim))
        return samples, VALS


            
def _plot_bestmodel_residual(OBJ_NAME, modelbox, objectbox, residuals, save_name=None,  **kwargs): 

    #kwargs: xlim, ylim, label
    maxx = 2*np.max(modelbox.flux)
    ylim = kwargs.get('ylim', (-1.15e-16, maxx))
    xlim = kwargs.get('xlim', (0.8, 2.5))
    label = kwargs.get('label', 'TripleSpec')

    fig = plot_spectrum(boxes=[modelbox, objectbox],
                filters=objectbox.filters,
                residuals=residuals,
                plot_kwargs=[{'ls': '-', 'lw': 1., 'color': 'black'}, 
                                {OBJ_NAME:{'ls':'-', 'color':'blue', 'label':label}}],
                xlim=xlim,
                ylim=ylim,
                object_type='star',
                ylim_res=(-10., 10.),
                scale=('linear', 'linear'),
                offset=(-0.4, -0.05),
                legend=[{'loc': 'lower left', 'frameon': False, 'fontsize': 11.},
                        {'loc': 'upper right', 'frameon': False, 'fontsize': 12.}],
                figsize=(8., 4.),
                quantity='flux density',
                output=f'{save_name}_mcmc_bestfit')
    res_x = residuals.spectrum[OBJ_NAME][:,0]
    res_y = residuals.spectrum[OBJ_NAME][:,1]
    J_res = np.where((res_x > 0.95) & (res_x < 1.4))
    H_res = np.where((res_x > 1.4)& (res_x < 1.8))
    K_res = np.where((res_x > 1.9)& (res_x < 2.5))
    JH_res = np.where(((res_x > 0.95)& (res_x < 1.3)) |
        ((res_x > 1.4)& (res_x < 1.8)))
    print('continuum residuals for:')
    print('J band: ', np.median(res_y[J_res]))
    print(' H band: ', np.median(res_y[H_res]))
    print(' K band: ', np.median(res_y[K_res]))
    print('JH bands: ', np.median(res_y[JH_res]))
    return np.median(res_y[J_res]), np.median(res_y[H_res]), np.median(res_y[K_res]), np.median(res_y[JH_res])



def plot_bestmodel(object_path, model_path, save_name, **kwargs):
    # kwargs: wave_unit, flux_unit, ylim, fontsize, plot_waterbands, plot_HI, normalize
    figsize = kwargs.get('figsize', (10,3))
    wave_unit = kwargs.get('yunit', 'um')
    flux_unit = kwargs.get('yunit', 'W/(m2*um)')
    normalize_value = kwargs.get('normalize_value', 1e-16)
    maxx = np.max(np.loadtxt(model_path)[:,1]/normalize_value) + (np.min(np.loadtxt(model_path)[:,1]/normalize_value)/2)
    ylim = kwargs.get('ylim', [-1,maxx])
    xlim = kwargs.get('xlim', [0.9,2.5])
    ls = kwargs.get('fontsize', 12)
    
    label = kwargs.get('label', 'TripleSpec')

    obj = np.loadtxt(object_path)
    mod = np.loadtxt(model_path)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=150)
    spec = [obj, mod]
    for j in np.arange(2):

        Wave = spec[j][:,0]
        Flux = spec[j][:,1] 
        idx = np.where(np.isfinite(Flux))
        Wave = Wave[idx].astype('double')
        Flux = Flux[idx].astype('double')
        if j == 0:
            Error = spec[j][:,2] 
            Error = Error[idx].astype('double')

        idx1 = np.where((Wave >= 0.965) & (Wave < 2.47))
        Wave = Wave[idx1]
        Flux = Flux[idx1]
            
        IND2 = np.where((Wave >= 1.344)&(Wave <= 1.451))
        Wave[IND2] = np.nan
        Flux[IND2] = np.nan

        IND3 = np.where((Wave >= 1.80)&(Wave <= 1.9624))
        Wave[IND3] = np.nan
        Flux[IND3] = np.nan

        Wave = Wave*u.Unit(wave_unit)
        Flux = (Flux * u.Unit(flux_unit)).to(u.erg / u.s/ u.cm**2/u.AA)
        
        if j == 0:
            Error = Error[idx1]
            Error[IND2] = np.nan
            Error[IND3] = np.nan
            Error = (Error * u.Unit(flux_unit)).to(u.erg / u.s/ u.cm**2/u.AA)

            ax.plot(Wave[0:-10], Flux[0:-10]/normalize_value, color='blue', linewidth=0.75, zorder=-1, label=label)
            y  = (Flux[0:-10]/normalize_value).value
            yerr = (Error[0:-10]/normalize_value).value
            ax.fill_between(Wave[0:-10].value, y-yerr, y+yerr, color='blue', 
                            alpha=0.4, zorder=-1)

        else:
            ax.plot(Wave[0:-10], Flux[0:-10]/normalize_value, color='k',  zorder=0, 
                    label=r'BT SETTL 2015')

    if kwargs.get('plot_HI'):
        if (1.2822 > xlim[0]) & (1.2822 < xlim[1]):
            ax.axvline(1.2822, color='gray', ymin=0.82, ymax=0.86, linewidth=1, linestyle='-')
            ax.text(1.2822, 0.9, r'Pa$\beta$', color='k', ha='center', transform=ax.get_xaxis_transform() )
        if (1.0941 > xlim[0]) & (1.0941 < xlim[1]):
            ax.axvline(1.0941, color='gray', ymin=0.82, ymax=0.86, linewidth=1, linestyle='-')
            ax.text(1.094+0.02, 0.9, r'Pa$\gamma$', color='k', ha='center', transform=ax.get_xaxis_transform() )
        if (2.1661 > xlim[0]) & (2.1661 < xlim[1]):
            ax.axvline(2.1661, color='gray', ymin=0.82, ymax=0.86, linewidth=1, linestyle='-')
            ax.text(2.1655, 0.9, r'Br$\gamma$', color='k', ha='center', transform=ax.get_xaxis_transform() )
        if (1.0833 > xlim[0]) & (1.0833 < xlim[1]):
            ax.axvline(1.0833, color='gray', ymin=0.82, ymax=0.86, linewidth=1, linestyle='-')
            ax.text(1.0833-0.03, 0.9, r'He I', color='k', ha='center', transform=ax.get_xaxis_transform() )

    ax.tick_params(which='major', direction='in', labelsize=ls, right=True, top=True, length=7)
    ax.tick_params(which='minor', direction='in', labelsize=ls, right=True, top=True, length=5)
    ax.minorticks_on()

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    float_str = "{0:.4g}".format(normalize_value)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if base == '1':
            norm_latex = r"$10^{{{0}}}$".format(int(exponent))
        else:
            norm_latex = r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        norm_latex = float_str

    ax.set_ylabel('Flux (' + norm_latex + r' erg/s/cm$^2$/$\AA$)', fontsize=ls)
    if wave_unit == 'um':
        wave_unit_latex = 'μm'
    ax.set_xlabel(f'Wavelength ({wave_unit_latex})', fontsize=ls)
    
    if kwargs.get('plot_waterbands'):
        ax.axvspan(1.113, 1.156, color='whitesmoke', alpha=1, zorder=-1)
        ax.axvspan(1.35, 1.491, color='whitesmoke', alpha=1, zorder=1)
        ax.axvspan(1.764, 2.01, color='whitesmoke', alpha=1, zorder=1)  
    ax.legend(loc = 'lower left', fontsize=ls-3)
    
    plt.savefig(f'{save_name}_bestfit.pdf', dpi=150)
    plt.clf()
    # plt.show()









    