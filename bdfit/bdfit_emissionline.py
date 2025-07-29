from species import SpeciesInit
from species.data.database import Database
from species.fit.emission_line import EmissionLine
from species.plot.plot_mcmc import plot_posterior
from species.util.model_util import gaussian_spectrum
from species.read.read_object import ReadObject

import os
import corner

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy import units as u
from astropy import constants as const

from specutils.manipulation import extract_region
from specutils.spectra import SpectralRegion
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum

from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.polynomial import Polynomial1D
from astropy.nddata import StdDevUncertainty

print("""Before starting, manually update species==0.9.1.dev
      
    in species.fit.emission_line.EmissionLine.lnlike_ultranest() attribute:
        - line 1184 update to: 
            species_db.add_samples(
                    sampler="ultranest",
                    samples=samples,
                    ln_prob=ln_prob,
                    tag=tag,
                    modelpar=modelpar,
                    attr_dict=attr_dict,
                    bounds = bounds,
                    normal_prior = {}, fixed_param={})

    in species.fit.emission_line.EmissionLine.integrate_flux() attribute:
        - line 470: remove @typechecked
        - line 767: update line to: return (line_flux, line_error), (line_lum_mean, line_lum_std)
""")

def hydrogen_lines(line):
    hydrogen = {}
    hydrogen['HeI'] = 1.08330
    hydrogen['Brg'] = 2.16612
    hydrogen['Pab'] = 1.28216
    hydrogen['Pag'] = 1.09411
    return hydrogen[line]

class bdfit_emissionline:
    def __init__(self, OBJ_NAME, PARALLAX, PARALLAX_err, Av,  mass, radius, specfile_name, spec_res=2700., Rin=5*u.Rsun, save_name=None):
        SpeciesInit()
        self.OBJ_NAME = OBJ_NAME
        if save_name is not None:
            self.save_name = save_name 
        else:
            self.save_name = OBJ_NAME
        self.PARALLAX = PARALLAX
        self.PARALLAX_err = PARALLAX_err
        self.spec_res = float(spec_res)
        self.specfile_name = specfile_name
        self.Av = Av

        self.Radius = radius[0] 
        self.Radius_err = radius[1] 
        self.Mass = mass[0] 
        self.Mass_err = mass[1] 
        self.Rin = Rin

        self.database = Database()

        self.database.add_object(OBJ_NAME,
                            parallax=(PARALLAX, PARALLAX_err),
                            app_mag=None,
                            flux_density=None,
                            spectrum={OBJ_NAME: (specfile_name, None, spec_res),},
                            deredden=Av)
        
        self.param_dict = {}
    
    def Alcala_scaling(self, line, A=None, B=None, wave=None):
        HERE = os.path.dirname(os.path.abspath(__file__))
        fil = os.path.join(HERE, 'alcala2017_linear_fits.csv')
        alcala_lines = pd.read_csv(fil, comment='#')
        if line is None:
            a = A
            b = B
            wave=None
            if a is None:
                raise ValueError('must provide A and B')
        elif line in alcala_lines.Diagnostic.values: 
            a = alcala_lines['a'].loc[alcala_lines['Diagnostic']==line].values[0]
            b = alcala_lines['b'].loc[alcala_lines['Diagnostic']==line].values[0]
            aerr = alcala_lines['a_err'].loc[alcala_lines['Diagnostic']==line].values[0]
            berr = alcala_lines['b_err'].loc[alcala_lines['Diagnostic']==line].values[0]
            wave = alcala_lines['wavelength'].loc[alcala_lines['Diagnostic']==line].values[0]
        else:            
            raise ValueError(f'Line not found: please use either None or one of the following lines: {alcala_lines.Diagnostic.to_list()}')
        return a, aerr, b, berr
        
    def _sub_cont(self, species_emission_line, polyfit=3):
        spec = species_emission_line.spectrum
        center_wavelength = species_emission_line.lambda_rest
        spec_extract = Spectrum1D(
            flux=spec[:, 1] * u.W,
            spectral_axis=spec[:, 0] * u.um,
            uncertainty=StdDevUncertainty(spec[:, 2] * u.W),
        )
        xmin, xmax = spec_extract.spectral_axis[0], spec_extract.spectral_axis[-1]
        region_cont = [SpectralRegion((center_wavelength-0.003)*u.um, (center_wavelength+0.003)*u.um)]

        g1_fit = fit_generic_continuum(
            spec_extract,
            median_window=3,
            model=Polynomial1D(polyfit),
            fitter=LinearLSQFitter(), exclude_regions=region_cont
        )
        
        continuum_fit = g1_fit(spec_extract.spectral_axis)
        spec_cont_sub = spec_extract - continuum_fit
        species_emission_line.continuum_flux = continuum_fit / u.W

        plt.figure(figsize=(6, 6))
        gs = mpl.gridspec.GridSpec(2, 1)
        gs.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = ax1.twiny()
        ax4 = ax2.twiny()
        ax1.axvline(center_wavelength-0.003)
        ax1.axvline(center_wavelength+0.003)
        ax1.tick_params(axis="both", which="both", colors="black", labelcolor="black", direction="in", width=1, labelsize=12,top=False,bottom=True,left=True,right=True,labelbottom=False,)
        ax2.tick_params(axis="both", which="both", colors="black", labelcolor="black", direction="in", width=1, labelsize=12,top=False,bottom=True,left=True,right=True,labelbottom=False,)
        ax3.tick_params(axis="both", which="both", colors="black", labelcolor="black", direction="in", width=1, labelsize=12,top=True,bottom=False,left=True,right=True)
        ax4.tick_params(axis="both", which="both", colors="black", labelcolor="black", direction="in", width=1, labelsize=12,top=True,bottom=False,left=True,right=True, labeltop=False,)

        ax1.set_ylabel("Flux (W m$^{-2}$ µm$^{-1}$)", fontsize=16)
        ax2.set_xlabel("Wavelength (µm)", fontsize=16)
        ax2.set_ylabel("Flux (W m$^{-2}$ µm$^{-1}$)", fontsize=16)
        ax3.set_xlabel("Velocity (km s$^{-1}$)", fontsize=16)

        ax1.get_yaxis().set_label_coords(-0.1, 0.5)
        ax2.get_xaxis().set_label_coords(0.5, -0.1)
        ax2.get_yaxis().set_label_coords(-0.1, 0.5)
        ax3.get_xaxis().set_label_coords(0.5, 1.12)

        ax1.plot(
            spec_extract.spectral_axis,
            spec_extract.flux,
            color="black",
            label=species_emission_line.spec_name,
        )
        ax1.plot(
            spec_extract.spectral_axis,
            continuum_fit,
            color="tab:blue",
            label="SB Continuum fit",
        )

        ax2.plot(
            spec_cont_sub.spectral_axis,
            spec_cont_sub.flux,
            color="black",
            label="SB Continuum subtracted",
        )

        ax3.plot(species_emission_line.spec_vrad, spec_extract.flux, ls="-", lw=0.0)
        ax4.plot(species_emission_line.spec_vrad, spec_cont_sub.flux, ls="-", lw=0.0)

        ax1.legend(loc="upper right", frameon=False, fontsize=12.0)
        ax2.legend(loc="upper right", frameon=False, fontsize=12.0)

        print(" [DONE]")

       
        plt.show()
    
        # Overwrite original spectrum with continuum-subtracted spectrum
        species_emission_line.spectrum[:, 1] = spec_cont_sub.flux

        species_emission_line.continuum_check = True
        return species_emission_line

    def measure_line_luminosity(self, line, wavel_range, integrate_line=True, fit_line=True , bounds = None, verbose=True, polyfit=3):
        if line == 'HeI':
            lambda_rest = hydrogen_lines('HeI')
            line_analysis = EmissionLine(object_name=self.OBJ_NAME,
                                spec_name=self.OBJ_NAME,
                                hydrogen_line=None,
                                lambda_rest=lambda_rest,
                                wavel_range=wavel_range)
        else:
            line_analysis = EmissionLine(object_name=self.OBJ_NAME,
                                spec_name=self.OBJ_NAME,
                                hydrogen_line=line,
                                lambda_rest=None,
                                wavel_range=wavel_range)
                

        line_analysis = self._sub_cont(line_analysis, polyfit=polyfit)

        # line_analysis.subtract_continuum(poly_degree=polyfit,
        #                                 plot_filename=None)
        
        if integrate_line:
            lineflux, lineflux_err, logLline, logLlineerr = line_analysis.integrate_flux(wavel_int=self.line_range(line),
                                    interp_kind='linear',
                                    plot_filename=f'{self.save_name}_{line}_integrateline.pdf') 
            
    
        
        if fit_line:
            if bounds is None:
                bounds={'gauss_amplitude': (0., 1e-14)}

            line_analysis.fit_gaussian(tag=line,
                                min_num_live_points=100,
                                bounds=bounds,
                                output='ultranest',
                                plot_filename=f'{self.save_name}_{line}_fitting.pdf',
                                show_status=True)

            fig = plot_posterior(tag=line,
                            title_fmt='.2f',
                            offset=(-0.4, -0.35), object_type='star',
                            output= f'{self.save_name}_{line}_fitting_posterior.pdf')
            plt.show()
            best = self.database.get_median_sample(tag=line)

            amp = best['gauss_amplitude']
            mean = best['gauss_mean']
            sigma = best['gauss_sigma']
            model_param = {
            "gauss_amplitude": amp,
            "gauss_mean": mean,
            "gauss_sigma":sigma}
            best_model = gaussian_spectrum(wavel_range, model_param, spec_res=1e5, double_gaussian=False)

            if os.path.exists(f'{self.save_name}_{line}_contsub_spec.txt'):
                os.remove(f'{self.save_name}_{line}_contsub_spec.txt')
            np.savetxt(f'{self.save_name}_{line}_contsub_spec.txt', line_analysis.spectrum)
            if os.path.exists(f'{self.save_name}_{line}_bestfit_gaussian.txt'):
                os.remove(f'{self.save_name}_{line}_bestfit_gaussian.txt')
            np.savetxt(f'{self.save_name}_{line}_bestfit_gaussian.txt', np.c_[best_model.wavelength, best_model.flux])

            plt.figure()
            plt.plot(line_analysis.spectrum[:,0], line_analysis.spectrum[:,1], 'k')
            plt.plot(best_model.wavelength, best_model.flux, 'b')
            plt.xlabel('wavelength (μm)')
            plt.ylabel('flux (W/m2/um)')
            plt.show()

            lineflux = best['line_flux']
            logLline = best['log_line_lum']
        
            box = self.database.get_samples(tag=line)
            params = box.parameters
            samples = box.samples
            ndim = len(params)
            samples0 = samples.reshape((-1, ndim))

            idx = params.index('line_flux')
            best_val = best['line_flux']
            q_16 = corner.quantile(samples0[:, idx], [0.16])[0]
            lineflux_err = (best_val - q_16)

            idx = params.index('log_line_lum')
            best_val = best['log_line_lum']
            q_16 = corner.quantile(samples0[:, idx], [0.16])[0]
            logLlineerr = (best_val - q_16)
            
        
        Line_Flux = lineflux * u.W/(u.m**2.)
        Line_Flux_err =  lineflux_err* u.W/(u.m**2.)
        Line_Luminosity = (10**logLline)* u.Lsun
        Line_Luminosity_err = (((10**logLline) * logLlineerr)/0.434)* u.Lsun
        self.param_dict[f'{line}_Line_Flux'] = Line_Flux.value
        self.param_dict[f'{line}_Line_Flux_err'] =Line_Flux_err.value
        self.param_dict[f'{line}_Line_Luminosity'] = Line_Luminosity.value
        self.param_dict[f'{line}_Line_Luminosity_err'] = Line_Luminosity_err.value
        self.param_dict[f'LOG_{line}_Line_Luminosity'] = logLline
        self.param_dict[f'LOG_{line}_Line_Luminosity_err'] = logLlineerr

        if verbose:
            print('Line Flux: ', Line_Flux, '+/-', Line_Flux_err)
            print('Line Luminosity: ', Line_Luminosity, '+/-', Line_Luminosity_err)
            print('LOG Line Luminosity: ', logLline*u.Lsun, '+/-', logLlineerr*u.Lsun)
        

    def measure_accretion_luminosity(self, line, Lline=None, verbose=True):
        if f'{line}_Line_Luminosity' in self.param_dict:
            line_lum, line_lum_err = self.param_dict[f'{line}_Line_Luminosity'], self.param_dict[f'{line}_Line_Luminosity_err']
        elif Lline is not None:
            line_lum, line_lum_err = Lline
        else:
            raise ValueError('must run self.measure_line_luminosity() or provide Lline values.')
        
        if not isinstance(line_lum, u.Quantity):
            line_lum = line_lum*u.Lsun
        if not isinstance(line_lum_err, u.Quantity):
            line_lum_err = line_lum_err*u.Lsun

        # scaling relation
        a,a_err,b,b_err = self.Alcala_scaling(line)
        Lacc_Lsun = 10**(a * np.log10(line_lum.value) + b)*u.Lsun

        #uncertainty propagation
        log10L = np.log10(line_lum.value)
        sigma_log10L = 0.434 * line_lum_err.value/line_lum.value

        alog10L = a * log10L
        alog10Lb = a * log10L + b 

        logsigma_Lacc = np.sqrt( b_err**2. + (alog10L * np.sqrt( (a_err/a)**2. + (sigma_log10L/log10L)**2. ))**2.) 

        sigma_Lacc = 2.303 * (10**(alog10Lb)) * logsigma_Lacc
        Lacc_Lsun_err = sigma_Lacc * u.Lsun

        if verbose:
            if not np.isnan(Lacc_Lsun_err.value):
                print('LOG Lacc_Lsun: ', np.log10(Lacc_Lsun.value)*Lacc_Lsun.unit, '+/-', logsigma_Lacc*Lacc_Lsun.unit)
                print('Lacc_Lsun: ', Lacc_Lsun, '+/-', Lacc_Lsun_err)
            else:
                print('LOG Lacc_Lsun: <', np.log10(Lacc_Lsun.value)*Lacc_Lsun.unit)
                print('Lacc_Lsun: <', Lacc_Lsun)
        self.param_dict[f'{line}_Accretion_Luminosity'] = Lacc_Lsun.value
        self.param_dict[f'{line}_Accretion_Luminosity_err'] = Lacc_Lsun_err.value
        self.param_dict[f'LOG_{line}_Accretion_Luminosity'] = np.log10(Lacc_Lsun.value)
        self.param_dict[f'LOG_{line}_Accretion_Luminosity_err'] = logsigma_Lacc
    
    def measure_Mdot(self, line, Lacc=None, verbose=True):
        if f'{line}_Accretion_Luminosity' in self.param_dict:
            Lacc, Lacc_err  = self.param_dict[f'{line}_Accretion_Luminosity'], self.param_dict[f'{line}_Accretion_Luminosity_err']
        elif Lacc is not None:
            Lacc, Lacc_err = Lacc
        else:
            raise ValueError('must run self.measure_accretion_luminosity() or provide Lacc values.')
        
        if not isinstance(Lacc, u.Quantity):
            Lacc = Lacc*u.Lsun
        if not isinstance(Lacc_err, u.Quantity):
            Lacc_err = Lacc_err*u.Lsun

        Mdot = (((1-(self.Radius/self.Rin))**-1) * (self.Radius*Lacc)/(const.G*self.Mass)).to(u.Msun/u.yr)
        Mdot_err = (Mdot * np.sqrt((Lacc_err/Lacc)**2. + (self.Radius_err/self.Radius)**2. + (self.Mass_err/self.Mass)**2.)).to(u.Msun/u.yr)

        LOGmass = (np.log10(self.Mass.value)* self.Mass.unit).to(u.Msun)
        LOGmass_err = (0.434 * self.Mass_err.value/self.Mass.value * self.Mass.unit).to(u.Msun)

        LOGMdot = (np.log10(Mdot.value) * Mdot.unit)
        LOGMdot_err = 0.434 * Mdot_err/Mdot

        if verbose:
            if not np.isnan(Mdot_err.value):
                print('Mdot Msun/yr:', Mdot, '+/-',Mdot_err)
                print()
                print('Log(Mdot): ', LOGMdot , '+/-',LOGMdot_err )
            else:
                print('Mdot: <', Mdot)
                print()
                print('Log(Mdot): <', LOGMdot)
            print('Log(Mass):', LOGmass, '+/-', LOGmass_err)
        self.param_dict[f'{line}_Accretion_Rate'] = Mdot.value
        self.param_dict[f'{line}_Accretion_Rate_err'] = Mdot_err.value
        self.param_dict[f'LOG_{line}_Accretion_Rate'] = LOGMdot.value
        self.param_dict[f'LOG_{line}_Accretion_Rate_err'] = LOGMdot_err.value
        self.param_dict[f'LOG_Mass'] = LOGmass.value
        self.param_dict[f'LOG_Mass_err'] = LOGmass_err.value


    def line_range(self, line):
        d = {'Pag':(1.0935, 1.095), 'Pab':(1.281, 1.2834), 'Brg':(2.165, 2.167)}
        return d[line]

    
    def run_measure_object(self, line, wavel_range, fit_line=True, integrate_line=False, upper_limit=False, bounds=None, deltalambda=None, verbose=True, polyfit=3):
        if upper_limit:
            self.upper_limit_derivation(line, wavel_range, deltalambda, verbose=verbose)
        else:
            self.measure_line_luminosity(line, wavel_range, fit_line=fit_line, integrate_line=integrate_line, bounds=bounds, verbose=verbose, polyfit=polyfit)

            self.measure_accretion_luminosity(line, verbose=verbose)
            
            self.measure_Mdot(line, verbose=verbose)
        

    
    def save_parameters(self):

        print(pd.DataFrame.from_dict(self.param_dict, orient='index', columns=['Value']))

        df = pd.DataFrame([self.param_dict])
        print('save best fit parameters to: ', f'{self.save_name}_emissionline_fitting.csv')
        df.to_csv(f'{self.save_name}_emissionline_fitting.csv', index=False)  


    def upper_limit_derivation(self, line, wavel_range, deltalambda, verbose=True):

        line_analysis = EmissionLine(object_name=self.OBJ_NAME,
                    spec_name=self.OBJ_NAME,
                    hydrogen_line=line,
                    lambda_rest=None,
                    wavel_range=wavel_range)
                
        line_analysis.subtract_continuum(poly_degree=3,
                                        plot_filename=None)
        
        if os.path.exists(f'{self.save_name}_{line}_bestfit_gaussian.txt'):
            os.remove(f'{self.save_name}_{line}_bestfit_gaussian.txt')
        if os.path.exists(f'{self.save_name}_{line}_contsub_spec.txt'):
            os.remove(f'{self.save_name}_{line}_contsub_spec.txt')
        np.savetxt(f'{self.save_name}_{line}_contsub_spec.txt', line_analysis.spectrum)
        
        wave = line_analysis.spectrum[:, 0]*u.um
        contsub_flux = line_analysis.spectrum[:, 1]*u.W/u.m**2./u.s


        spec = Spectrum1D(spectral_axis = wave, flux=contsub_flux)

        region = SpectralRegion(wavel_range[0]*u.um, wavel_range[1]*u.um)
        SS2 = extract_region(spec, region)   
        SS2flux = SS2.flux.value 
        local_cont_mean = np.nanmean(SS2flux)
        local_cont_std = np.nanstd(SS2flux)/10000

        real_center_wavelength = hydrogen_lines(line) * u.um
        region_line = SpectralRegion((real_center_wavelength-0.00035*u.um), (real_center_wavelength+0.00035*u.um))
        len_line = extract_region(spec, region_line)    

        LineFlux_upp = 3 * local_cont_std * deltalambda * np.sqrt(len(len_line.flux)) #W/m2

        LineLum_upp = (
                4.0
                * np.pi
                * (1e3 * const.pc.value / self.PARALLAX) ** 2
                * LineFlux_upp
            )
        LineLum_upp /= const.L_sun.value  # (Lsun)

        a,a_err,b,b_err = self.Alcala_scaling(line)
        Lacc_upp = 10**(a * np.log10(LineLum_upp) + b) * u.Lsun
        print(self.Radius, self.Rin)
        Mdot_upp = (((1-(self.Radius/self.Rin))**-1) * (self.Radius*Lacc_upp)/(const.G*self.Mass)).to(u.Msun/u.yr)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams["axes.axisbelow"] = False

        plt.figure(figsize=(6, 3))
        gs = mpl.gridspec.GridSpec(1, 1)
        gs.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = ax1.twiny()

        ax1.plot(wave, contsub_flux, color='k')
        ax1.axvline((real_center_wavelength-(0.00035*u.um)).value, color='gray', linestyle=':')
        ax1.axvline((real_center_wavelength+(0.00035*u.um)).value, color='gray', linestyle=':')
        ax1.axhline(0, color='orange')
        spec_vrad = (1e-3 * const.c.value * (wave - real_center_wavelength) / real_center_wavelength )
        ax2.plot(spec_vrad, contsub_flux, ls="-", lw=0.0)

        ax1.tick_params(axis="both", which="major", colors="black",
                        labelcolor="black", direction="in", width=1, length=5,
                        labelsize=12, top=False, bottom=True, left=True,right=True)
        ax1.tick_params(axis="both", which="minor", colors="black",
                        labelcolor="black", direction="in", width=1, length=3,
                        labelsize=12, top=False, bottom=True, left=True,right=True)
        ax2.tick_params(axis="both", which="major", colors="black",
                        labelcolor="black", direction="in", width=1, length=5,
                        labelsize=12, top=True, bottom=False, left=False,right=True)
        ax2.tick_params(axis="both", which="minor", colors="black",
                        labelcolor="black", direction="in", width=1, length=3,
                        labelsize=12, top=True, bottom=False, left=False,right=True)
        
        ax1.set_xlabel("Wavelength (µm)", fontsize=16)
        ax1.set_ylabel("Flux (W m$^{-2}$ µm$^{-1}$)", fontsize=16)
        ax2.set_xlabel("Velocity (km s$^{-1}$)", fontsize=16)

        ax1.get_xaxis().set_label_coords(0.5, -0.12)
        ax1.get_yaxis().set_label_coords(-0.1, 0.5)
        ax2.get_xaxis().set_label_coords(0.5, 1.12)
        plt.show()

        if verbose:
            print('Line Flux < ', LineFlux_upp*u.W/u.m**2.)
            print('Line Luminosity < ', LineLum_upp*u.Lsun)
            print('LOG Line Luminosity < ', np.log10(LineLum_upp)*u.Lsun)
            print('Accretion Luminosity < ', Lacc_upp)
            print('LOG Accretion Luminosity < ', np.log10(Lacc_upp.value)*u.Lsun)
            print('Accretion Rate < ', Mdot_upp)
            print('LOG Accretion Rate < ', np.log10(Mdot_upp.value)*Mdot_upp.unit)

        LOGmass = (np.log10(self.Mass.value)* self.Mass.unit).to(u.Msun)
        LOGmass_err = (0.434 * self.Mass_err.value/self.Mass.value * self.Mass.unit).to(u.Msun)

        self.param_dict[f'{line}_Line_Flux'] = LineFlux_upp
        self.param_dict[f'{line}_Line_Flux_err'] =np.nan
        self.param_dict[f'{line}_Line_Luminosity'] = LineLum_upp
        self.param_dict[f'{line}_Line_Luminosity_err'] = np.nan
        self.param_dict[f'LOG_{line}_Line_Luminosity'] = np.log10(LineLum_upp)
        self.param_dict[f'LOG_{line}_Line_Luminosity_err'] = np.nan


        self.param_dict[f'{line}_Accretion_Luminosity'] = Lacc_upp.value
        self.param_dict[f'{line}_Accretion_Luminosity_err'] = np.nan
        self.param_dict[f'LOG_{line}_Accretion_Luminosity'] = np.log10(Lacc_upp.value)
        self.param_dict[f'LOG_{line}_Accretion_Luminosity_err'] = np.nan
    
        self.param_dict[f'{line}_Accretion_Rate'] = Mdot_upp.value
        self.param_dict[f'{line}_Accretion_Rate_err'] = np.nan
        self.param_dict[f'LOG_{line}_Accretion_Rate'] = np.log10(Mdot_upp.value)
        self.param_dict[f'LOG_{line}_Accretion_Rate_err'] = np.nan
        self.param_dict[f'LOG_Mass'] = LOGmass.value
        self.param_dict[f'LOG_Mass_err'] = LOGmass_err.value

from astropy.io import fits
def indiv_plots(fil, PaBspec, PaGspec, BrGspec, xunit, yunit, fs=12,  
                PaBmodel=None, PaGmodel=None, BrGmodel=None, 
                savefig=True, **kwargs):

    data1 = fits.open(fil)[0]
    wave1 = data1.data[0] * u.Unit(xunit)
    flux1 = data1.data[1] * u.Unit(yunit)
    err1 = data1.data[2] * u.Unit(yunit)

    flux1 = flux1.to(u.erg / u.s/ u.cm**2/u.AA)
    fluxnorm = flux1.value/1e-16

    err1 = err1.to(u.erg / u.s/ u.cm**2/u.AA)
    errnorm = err1.value/1e-16
    
    fig = plt.figure(constrained_layout=True, figsize=(8,5), dpi=150)
    gs = fig.add_gridspec(nrows=2,ncols=3, height_ratios=[1,1])
    ax = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    ax.plot(wave1, fluxnorm, 'k')
    
    lines = ['Pab', 'Pag', 'Brg', 'HeI']
    xlabel = [1.2822, 1.094+0.02, 2.1655, 1.0833-0.04]
    text = [r'Pa$\beta$', r'Pa$\gamma$', r'Br$\gamma$', 'He I']
    for i in np.arange(4):
        ax.axvline(hydrogen_lines(lines[i]), color='gray', ymin=0.82, ymax=0.86, linewidth=1, linestyle='-')
        ax.text(xlabel[i], 0.89, text[i], color='k', ha='center', transform=ax.get_xaxis_transform() )

    ax.tick_params(which='major', direction='in', labelsize=fs, right=True, top=True, length=7)
    ax.tick_params(which='minor', direction='in', labelsize=fs, right=True, top=True, length=5)
    ax.minorticks_on()

    ylimspec = kwargs.get('ylimspec', [0,10])
    ax.set_xlim(0.92, 2.5)
    ax.set_ylim(ylimspec[0], ylimspec[1])
    ax.set_ylabel(r'Flux (10$^{-16}$ erg/s/cm$^2$/$\AA$)', fontsize=fs)
    ax.set_xlabel('Wavelength (μm)', fontsize=fs)
    
    ax.axvspan(1.113, 1.156, color='gainsboro', alpha=0.3)
    ax.axvspan(1.333, 1.491, color='gainsboro', alpha=0.3)
    ax.axvspan(1.764, 2.0, color='gainsboro', alpha=0.3)  
    
    # ##############################################
    ax = [ax1, ax2, ax3]
    specs = [PaBspec, PaGspec, BrGspec]
    mods = [PaBmodel, PaGmodel, BrGmodel]
    for i in np.arange(3):
        real_center_wavelength = hydrogen_lines(lines[i]) * u.um
        spec = np.loadtxt(specs[i])
        if mods[i] is not None:
            mod = np.loadtxt(mods[i])
            flux = (mod[:,1]*u.W/u.m**2./u.um).to(u.erg / u.s/ u.cm**2/u.AA)
            ax[i].plot(mod[:,0], flux/1e-16, color='blue')
        wave = spec[:,0] * u.um
        flux = (spec[:,1]*u.W/u.m**2./u.um).to(u.erg / u.s/ u.cm**2/u.AA)
        ax[i].step(spec[:,0], flux/1e-16, color='k')
        ax[i].set_ylim(top=np.max(flux.value) * 1.5)

        axT = ax[i].twiny()
        c = 2.99e5 # speed of light in km/s
        delta_lambda = (wave - (real_center_wavelength))/(real_center_wavelength)    # the fractional wavelength difference
        velocity_array = c * delta_lambda
    
        IND = np.where(abs(velocity_array)<=1000)
        newxmin = wave[IND].value[0]
        newxmax = wave[IND].value[-1]
        ax[i].set_xlim([newxmin, newxmax])
        min_x, max_x = -1000, 1000#velocity_array[0], velocity_array[-1]
        axT.set_xlim([min_x, max_x])
        axT.tick_params(which='major', direction='in', labelsize=fs, right=False, top=True, length=7, bottom=False, left=False)
        axT.tick_params(which='minor', direction='in', labelsize=fs, right=False, top=True, length=5, bottom=False, left=False)
        axT.minorticks_on()
        axT.set_xlabel('velocity (km/s)', fontsize=fs)

        ax[i].tick_params(which='major', direction='in', labelsize=fs, right=True, top=False, length=7, bottom=True, left=True)
        ax[i].tick_params(which='minor', direction='in', labelsize=fs, right=True, top=False, length=5, bottom=True, left=True)
        ax[i].minorticks_on()

        ax[i].text(0.1, 0.85, text[i], transform=ax[i].transAxes, fontsize=fs+2 )
        ylimlines = kwargs.get('ylimlines', [-0.5, 4])
        ax[i].set_ylim(ylimlines[0], ylimlines[1])
        ax[i].set_xlabel('wavelength (μm)', fontsize=fs)

    ax1.set_ylabel('Continuum-subtracted Flux\n(10$^{-16}$ erg/s/cm$^2$/$\AA$)', fontsize=fs)
        
    # extra_title = kwargs.get('extra_title', name)
    # ax.set_title(extra_title, fontsize=ls+4)
    # if savefig:
    #     plt.savefig(maindir + f'plots/NIRAccretion_plots/{extra_title}.pdf', dpi=150)
    plt.show()
        

