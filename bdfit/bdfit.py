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


class fitspec:
    def __init__(self, OBJ_NAME, PARALLAX, PARALLAX_err, specfile_full_name,
                 specfile_short_name = None, save_name = None, 
                 spec_res=500., model='bt-settl-cifist'):
        SpeciesInit()
        self.OBJ_NAME = OBJ_NAME
        if save_name is not None:
            self.savename = save_name 
        else:
            self.savename = OBJ_NAME
        self.PARALLAX = PARALLAX
        self.PARALLAX_err = PARALLAX_err
        self.spec_res = float(spec_res)
        self.model=model
        self.specfile_short_name = specfile_short_name
        self.specfile_full_name = specfile_full_name

        self.database = Database()
        if self.specfile_short_name is not None:
            if os.path.exists(self.specfile_short_name):
                self.database.add_object(OBJ_NAME,
                                    parallax=(PARALLAX, PARALLAX_err),
                                    app_mag=None,
                                    flux_density=None,
                                    spectrum={f'{OBJ_NAME}_short': (self.specfile_short_name, None, spec_res),
                                            f'{OBJ_NAME}_full': (self.specfile_full_name, None, spec_res),},
                                    deredden=None)
            else:
                raise ValueError(f'{self.specfile_short_name} does not exist. Please check your file name')
        else:
            self.database.add_object(OBJ_NAME,
                                parallax=(PARALLAX, PARALLAX_err),
                                app_mag=None,
                                flux_density=None,
                                spectrum={f'{OBJ_NAME}_full': (self.specfile_full_name, None, spec_res),},
                                deredden=None)
        
    def compare_empirical_library(self, library, spec_length='short', wavel_range=(0.9,1.9),sptypes=['M'],  av_range =np.arange(0,5, 0.1), **kwargs):
        #kwargs: xlim, ylim, flux_offset, label_pos, figsize
        figsize=kwargs.get('figsize', (4, 3))
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        flux_offset = kwargs.get('flux_offset', 3e-15)
        label_pos = kwargs.get('label_position', (1.6, 1.5e-14))

        self.database.add_spectra(spec_library=library, sptypes=sptypes)
        compare = CompareSpectra(object_name=self.OBJ_NAME,
                    spec_name=[self.OBJ_NAME + '_' + spec_length])
        
        compare.spectral_type(tag=self.OBJ_NAME,
                     spec_library=library,
                     wavel_range=wavel_range,
                     sptypes=sptypes, 
                     av_ext=av_range, )

        fig = plot_empirical_spectra(tag=self.OBJ_NAME,
                                n_spectra=8,
                                flux_offset=flux_offset,
                                label_pos=label_pos,
                                xlim=xlim,
                                ylim=ylim,
                                title=None,
                                offset=None,
                                figsize=figsize,
                                output=f'{self.savename}_{library}_comparison')
        
    def compare_model(self, spec_length='short', wavel_range=(0.9, 1.9), teff_range=(2100., 2900.), av_range=np.arange(0,5, 0.1), **kwargs):
        #kwargs:  xlim, ylim, flux_offset, label_pos, figsize

        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        figsize=kwargs.get('figsize', (4, 3))
        flux_offset = kwargs.get('flux_offset', 3e-15)
        label_pos = kwargs.get('label_position', (1.6, 1.5e-14))

        self.database.add_model(model=self.model, teff_range=teff_range, 
                  wavel_range=wavel_range)
        
        compare = CompareSpectra(object_name=self.OBJ_NAME,
                    spec_name=[self.OBJ_NAME + '_' + spec_length])

        compare.compare_model(tag=self.OBJ_NAME,
                            model=self.model,
                            av_points=av_range,
                            fix_logg=4.0,
                            scale_spec=None,
                            weights=False,
                            inc_phot=False)
                            
        fig = plot_model_spectra(tag=self.OBJ_NAME,
                                n_spectra=5,
                                flux_offset=flux_offset,
                                label_pos=label_pos,
                                xlim=xlim,
                                ylim=ylim,
                                title=None,
                                offset=None,
                                figsize=figsize,
                                output=f'{self.savename}_btsettlcifistgrid_comparison',
                                leg_param=['teff', 'logg', 'ism_ext', 'radius'])
        
    def mcmc_fitting(self, bounds, spec_length='short', n_live_points=2000, teff_range=(2100., 3600.)):
        if ('teff' not in bounds) | ('radius' not in bounds) | ('ext_av' not in bounds )  | ('logg' not in bounds):
            raise ValueError('bounds should include "teff", "radius", "ext_av", "logg".  To exclude from fitting, make range equal to each other.  ex "logg":(4.0, 4.0)')
        
        self.database.add_model(model=self.model, teff_range=teff_range)

        fit = FitModel(object_name=self.OBJ_NAME,
               model=self.model,
               bounds=bounds,
               inc_phot=False,
               inc_spec=[self.OBJ_NAME + '_' + spec_length],
               fit_corr=None,
               apply_weights=False, ext_model='CCM89') 
        
        fit.run_multinest(tag=self.OBJ_NAME,
                  n_live_points=n_live_points,
                  resume=False,
                  output='multinest/',
                  kwargs_multinest=None)
        
    def posterior(self):
        fig = plot_posterior(tag=self.OBJ_NAME,
                     offset=(-0.3 , -0.3),
                     title_fmt='.2f',
                     inc_luminosity=True,
                     inc_mass=False,
                     output=f'{self.savename}_mcmc_posterior')
        
    def bestfit_samples(self):
        box_plotting = self.database.get_samples(tag=self.OBJ_NAME)
        samples_plotting = box_plotting.samples
        params_plotting = box_plotting.parameters
        
        teff_index = np.argwhere(np.array(params_plotting) == "teff")[0]
        radius_index = np.argwhere(np.array(params_plotting) == "radius")[0]
        R_JUP = constants.R_jup.value
        L_SUN = constants.L_sun.value
        SIGMA_SB = constants.sigma_sb.value
        lum_atm_plotting = (4.0 * np.pi * (samples_plotting[..., radius_index] * R_JUP) ** 2 * SIGMA_SB * samples_plotting[..., teff_index] ** 4.0 / L_SUN)

        samples_plotting = samples_plotting[:, 0:4]
        params_plotting = params_plotting[0:4]
        ndim_plotting = len(params_plotting)
  
        samples_plotting = np.append(samples_plotting, np.log10(lum_atm_plotting), axis=-1)
        params_plotting.append("log_lum_atm")
        ndim_plotting += 1
        self.best_samples = samples_plotting.reshape((-1, ndim_plotting))
        return self.best_samples
    
    def all_samples(self, spec_length='full'):
        self.samples = self.database.get_mcmc_spectra(tag=self.OBJ_NAME,
                                random=30,
                                wavel_range=None,
                                spec_res=self.spec_res)

        self.best = self.database.get_median_sample(tag=self.OBJ_NAME)
        read_model = ReadModel(model=self.model, wavel_range=None)
        self.modelbox = read_model.get_model(model_param=self.best,
                                        spec_res=self.spec_res)
        objectbox = self.database.get_object(object_name=self.OBJ_NAME,
                                        inc_phot=False,
                                        inc_spec=[self.OBJ_NAME + '_' + spec_length])
        self.objectbox = update_objectbox(objectbox=objectbox, model_param=self.best)
        self.residuals = get_residuals(tag = self.OBJ_NAME,
                                       parameters=self.best,
                                       objectbox=objectbox,
                                       inc_phot=False,
                                       inc_spec=True)
        return self.samples, self.best, self.modelbox, self.objectbox, self.residuals 
    
    def plot_bestmodel_residual(self, spec_length = 'full', **kwargs): 
        #kwargs: xlim, ylim, label
        ylim = kwargs.get('ylim', (-1.15e-16, 3e-14))
        xlim = kwargs.get('xlim', (0.8, 2.5))
        label = kwargs.get('label', 'TripleSpec')
   
        fig = plot_spectrum(boxes=[self.samples, self.modelbox, self.objectbox],
                    filters=self.objectbox.filters,
                    residuals=self.residuals,
                    plot_kwargs=[{'ls': '-', 'lw': 0.2, 'color': 'gray'},
                                 {'ls': '-', 'lw': 1., 'color': 'black'}, 
                                 {self.OBJ_NAME + '_' + spec_length:{'ls':'-', 'color':'blue', 'label':label}}],
                    xlim=xlim,
                    ylim=ylim,
                    ylim_res=(-10., 10.),
                    scale=('linear', 'linear'),
                    offset=(-0.4, -0.05),
                    legend=[{'loc': 'lower left', 'frameon': False, 'fontsize': 11.},
                            {'loc': 'upper right', 'frameon': False, 'fontsize': 12.}],
                    figsize=(8., 4.),
                    quantity='flux density',
                    output=f'{self.savename}_mcmc_bestfit')
        res_x = self.residuals.spectrum[self.OBJ_NAME + '_' + spec_length][:,0]
        res_y = self.residuals.spectrum[self.OBJ_NAME + '_' + spec_length][:,1]
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
    

    
    def plot_bestmodel(self, obj, mod, spec_length='full', **kwargs):
        # kwargs: wave_unit, flux_unit, ylim, fontsize, plot_waterbands, plot_HI, normalize
        figsize = kwargs.get('figsize', (10,3))
        wave_unit = kwargs.get('yunit', 'um')
        flux_unit = kwargs.get('yunit', 'W/(m2*um)')
        ylim = kwargs.get('ylim', [-1,10])
        xlim = kwargs.get('xlim', [0.9,2.5])
        ls = kwargs.get('fontsize', 12)
        normalize_value = kwargs.get('normalize_value', 1e-16)
        label = kwargs.get('label', 'TripleSpec')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=150)
        spec = [obj, mod]
        for j in np.arange(2):
            if j == 0:
                Wave = spec[j].spectrum[self.OBJ_NAME + '_' + spec_length][0][:,0]
                Flux = spec[j].spectrum[self.OBJ_NAME + '_' + spec_length][0][:,1] 
                Error = spec[j].spectrum[self.OBJ_NAME + '_' + spec_length][0][:,2] 
                idx = np.where(np.isfinite(Flux))
                Wave = Wave[idx].astype('double')
                Flux = Flux[idx].astype('double')
                Error = Error[idx].astype('double')
            else:
                Wave = spec[j].wavelength
                Flux = spec[j].flux
                idx = np.where(np.isfinite(Flux))
                Wave = Wave[idx].astype('double')
                Flux =Flux[idx].astype('double')

                
                leg_vals = spec[j].parameters

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
                        label=r'BT SETTL 2015: $T_{\rm eff}$=' + f'{int(leg_vals["teff"])} K, log g = {round(leg_vals["logg"],1)}' + r', A$_{\rm V}$=' + f'{round(leg_vals["ext_av"],1)} mag')

        if kwargs.get('plot_HI'):
            ax.axvline(1.2822, color='gray', ymin=0.82, ymax=0.86, linewidth=1, linestyle='-')
            ax.axvline(1.0941, color='gray', ymin=0.82, ymax=0.86, linewidth=1, linestyle='-')
            ax.axvline(2.1661, color='gray', ymin=0.82, ymax=0.86, linewidth=1, linestyle='-')
            ax.axvline(1.0833, color='gray', ymin=0.82, ymax=0.86, linewidth=1, linestyle='-')
            ax.text(1.2822, 0.9, r'Pa$\beta$', color='k', ha='center', transform=ax.get_xaxis_transform() )
            ax.text(1.094+0.02, 0.9, r'Pa$\gamma$', color='k', ha='center', transform=ax.get_xaxis_transform() )
            ax.text(2.1655, 0.9, r'Br$\gamma$', color='k', ha='center', transform=ax.get_xaxis_transform() )
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
        
        # from synphot.blackbody import BlackBody1D
        # from synphot.units import FLAM
        # bb = BlackBody1D(temperature=290*u.K)
        # wav = Wave[0:-10].to(u.AA)
        # wav2 = wav[np.isfinite(wav)]
        # flux = bb(wav2).to(u.erg/u.s/u.cm**2./u.AA, u.spectral_density(wav2))

        # fluxorig = Flux[0:-10]
        # fluxdisk = 5e-8*flux
        # newy = fluxorig[np.isfinite(wav)] + fluxdisk
        
        # ax.plot(wav2.to(u.um), fluxdisk/1e-16, 'k')
        # ax.plot(wav2.to(u.um), newy/1e-16, color='r')
        
        ##############################################

        plt.savefig(f'{self.savename}_bestfit.pdf', dpi=150)
        plt.show()

    def _compute_individual_error(self, bounds, spec_length='short', n_live_points=2000, teff_range=(2100., 3600.)):

        self.mcmc_fitting(bounds, spec_length=spec_length, n_live_points=n_live_points, teff_range=teff_range)

        box = self.database.get_samples(tag=self.OBJ_NAME)
        samples = box.samples
        params = box.parameters
    
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
        
        samples = box.samples[:, 0:3]
        params = box.parameters[0:3]
        ndim = len(params)

        samples = np.append(samples, np.log10(lum_atm), axis=-1)
        params.append("log_lum_atm")
        ndim += 1
        samples0 = samples.reshape((-1, ndim))

        
        VALS = {}
        VALS['teff'] = [self.best['teff'], 100]
        for i in np.arange(ndim):
            q_16, q_50, q_84 = corner.quantile(samples0[:, i], [0.16, 0.5, 0.84])

            best_val = self.best[params[i]]
            if best_val > q_50:
                err = best_val - q_16  
            else:
                err = q_84 - best_val
            VALS[params[i]] = [best_val, err]
            print(params[i], best_val, err)
            
        samples = np.append(np.ones_like(np.log10(lum_atm)), samples, axis=-1)
        params.insert(0, 'teff')
        ndim += 1
        samples = samples.reshape((-1, ndim))
        return samples, VALS
    

    def compute_errors(self, bounds, spec_length='short', Teff=None, teff_range=(2100., 3600.), n_live_points=2000):
        if Teff is None:
            Teff = self.modelbox.parameters["teff"]

        self.database.add_model(model=self.model, teff_range=teff_range)
        Teff_plus_offset = Teff+100.
        Teff_minus_offset = Teff-100.

        bounds_pos = bounds.copy()
        bounds_min = bounds.copy()

        if (Teff_plus_offset != bounds['teff'][0]) & (Teff_plus_offset != bounds['teff'][1]):
            print(f'>>> changing bound["teff"] from {bounds["teff"]} to {(Teff_plus_offset, Teff_plus_offset)}')

        bounds_pos['teff'] = (Teff_plus_offset, Teff_plus_offset)
        plus_samples, plus_vals = self._compute_individual_error(bounds_pos, spec_length=spec_length, n_live_points=n_live_points, teff_range=teff_range)

        if (Teff_minus_offset != bounds['teff'][0]) & (Teff_minus_offset != bounds['teff'][1]):
            print(f'changing bound["teff"] from {bounds["teff"]} to {(Teff_minus_offset, Teff_minus_offset)}')
        bounds_min['teff'] = (Teff_minus_offset, Teff_minus_offset)
        minus_samples, minus_vals = self._compute_individual_error(bounds_min, spec_length=spec_length, n_live_points=n_live_points, teff_range=teff_range)
        
        return plus_samples, plus_vals, minus_samples, minus_vals

    def plot_error_corner(self, plus_samples, plus_vals, minus_samples, minus_vals, Teff=None):
        if Teff is None:
            Teff = self.modelbox.parameters["teff"]
        print('best fit Teff = ', Teff)
        
        red = '#D81B60'
        blue = '#1E88E5'
        labels=[r"$T_{\rm eff}$ (K)", r"$R_p\ (R_J)$", r"parallax (mas)", r"$A_V$ (mag)", r"$\log L_p / L_\odot$"]
        figure = corner.corner(self.best_samples, labels=labels, quantiles=[0.5])

        axes = np.array(figure.axes).reshape((5, 5))
        labels=[r"$T_{\rm eff}$", r"$R_p$", r"$\varpi$", r"$A_V$", r"$\log L_p / L_\odot$"]
        labelend = ["K", r"$R_J$", "mas", "mag", ""]
        for i, k in enumerate(list(plus_vals.keys())):
            ax = axes[i, i]
            if i > 0:
                ax.set_title(labels[i] + '$=$' + str(round(plus_vals[k][0],2)) + '$^{+' + str(round(plus_vals[k][1],2)) + '}_{-' + str(round(minus_vals[k][1],2)) + '}$ ' + labelend[i])
                if k == 'radius':
                    ax.axvline(plus_vals[k][0]-plus_vals[k][1], color=red, linestyle='--')
                    ax.axvline(plus_vals[k][0]+minus_vals[k][1], color=blue, linestyle='--')
                else:
                    ax.axvline(plus_vals[k][0]+plus_vals[k][1], color=red, linestyle='--')
                    ax.axvline(plus_vals[k][0]-minus_vals[k][1], color=blue, linestyle='--')
            

                ax.hist(plus_samples[:,i], bins=20, color=red, histtype='step')
                ax.hist(minus_samples[:,i], bins=20, color=blue, histtype='step')
                minn = min(min(plus_samples[:,i]), min(minus_samples[:,i]))
                maxx = max(max(plus_samples[:,i]), max(minus_samples[:,i]))
                ax.set_xlim(minn, maxx)
            else:
                ax.set_title(labels[i] + '$=$' + str(round(plus_vals[k][0],2)) + ' ' + labelend[i])


        for yi in np.arange(5):
            for xi in np.arange(yi):
                ax = axes[yi, xi]
                if (yi > 0) & (xi>0):
                    ax = axes[yi, xi]
                    corner.hist2d(plus_samples[:, xi],plus_samples[:, yi], ax=ax, color=red)
                    corner.hist2d(minus_samples[:, xi],minus_samples[:, yi], ax=ax, color=blue)
                    minn = min(min(plus_samples[:,xi]), min(minus_samples[:,xi]))
                    maxx = max(max(plus_samples[:,xi]), max(minus_samples[:,xi]))
                    ax.set_xlim(minn, maxx)
                    
                    minn = min(min(plus_samples[:,yi]), min(minus_samples[:,yi]))
                    maxx = max(max(plus_samples[:,yi]), max(minus_samples[:,yi]))
                    ax.set_ylim(minn, maxx)
                if (xi > 0):  
                    ax.set_yticklabels([])
                if yi <4:
                    ax.set_xticklabels([])
        
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
        plt.savefig(f'{self.savename}_posterior_error.pdf', dpi=150)
        plt.show()

    def save_bestfit_model(self):
        best = self.database.get_median_sample(tag=self.OBJ_NAME)
        read_model = ReadModel(model=self.model, wavel_range=None)
        modelbox = read_model.get_model(model_param=best,
                                        spec_res=self.spec_res)
    
        Wave = modelbox.wavelength
        Flux = modelbox.flux
        idx = np.where(np.isfinite(Flux))
        Wave = Wave[idx].astype('double')
        Flux =Flux[idx].astype('double')
                    
        final_model = np.c_[Wave, Flux]
        leg_vals = modelbox.parameters
        print('saving best fit model to: ', f'{self.savename}_BTSettl_bestfit_spectra_T{int(leg_vals["teff"])}_logg{round(leg_vals["logg"],1)}_Av{round(leg_vals["ext_av"],1)}.txt')
        np.savetxt(f'{self.savename}_BTSettl_bestfit_spectra_T{int(leg_vals["teff"])}_logg{round(leg_vals["logg"],1)}_Av{round(leg_vals["ext_av"],1)}.txt', final_model)

    def save_bestfit_parameters(self, plus_vals, minus_vals):
        name = []
        value = []
        minn = []
        maxx = []

        for key, val in plus_vals.items():
            if key != 'parallax':
                name.append(key)
                value.append(val[0])
                minn.append(minus_vals[key][1])
                maxx.append(val[1])
        d = {'name': name, 'best_fit': value, 'err_lo':minn, 'err_hi':maxx}
        df = pd.DataFrame(d)
        print('save best fit parameters to: ', f'{self.savename}_BTSettl_bestfit_parameters.csv')
        df.to_csv(f'{self.savename}_BTSettl_bestfit_parameters.csv')            
            
