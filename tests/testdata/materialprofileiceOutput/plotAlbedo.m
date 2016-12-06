clear all;

% Begin input

downFile = 'cosine_irradiance_downward.txt';
upFile = 'cosine_irradiance_upward.txt';
runNo    = 1;

% End input

dataDown = readIrradiance(downFile);
dataUp = readIrradiance(upFile);
albedo = dataUp(runNo).irradiance(:,:)./dataDown(runNo).irradiance(:,:);
plot(dataDown(runNo).wavelength, albedo,'linewidth',1);
hl = legend(num2str(dataDown(runNo).depth'),4);
%set(get(hl,'title'),'string','Depth [m]');
set(gca,'xminortick','on','yminortick','on')
set(gca,'ylim',[0 1.0])
grid on
xlabel('Wavelength [nm]')
ylabel('Albedo')
title('Upward irradiance divided by downward irradiance')