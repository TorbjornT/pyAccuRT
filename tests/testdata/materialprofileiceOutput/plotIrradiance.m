% This Matlab script plots the AccuRT computed irradiance versus depth
% and wavelength for a given run.

clear all;

% Begin input

fileName     = 'cosine_irradiance_total_downward.txt';
runNo        = 1;
plotProfiles = false; % Set true to plot irradiance depth profiles

% End input

data       = readIrradiance(fileName);
nStreams   = data(runNo).nStreams;
irradiance = data(runNo).irradiance;
depth      = data(runNo).depth;
wavelength = data(runNo).wavelength;
%tryme
if (plotProfiles)
  plot(irradiance,depth,'linewidth',1);
  set(gca,'ydir','reverse')
  hl = legend(num2str(wavelength'),4);
  %set(get(hl,'title'),'string','Wavelengths [nm]');
  % Total irradiance = diffuse + direct
  xlabel('Total irradiance [W m^{-2} nm^{-1}]')
  ylabel('Depth [m]')
else
  plot(wavelength,irradiance,'linewidth',1);
  hl = legend(num2str(depth'),3);
  %set(get(hl,'title'),'string','Depth [m]');
  xlabel('Wavelength [nm]')
  % Total irradiance = diffuse + direct
  ylabel('Total irradiance [W m^{-2} nm^{-1}]')
end

set(gca,'xminortick','on','yminortick','on')
grid on
title([strrep(fileName,'_','\_') ' using ' num2str(nStreams) ' streams'])
