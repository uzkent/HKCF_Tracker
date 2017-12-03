function [data_Sfinal]=NoiseAdd(data,ml)
%% THIS FUNCTION ADDS NOISE TO THE INPUT DATA CONSIDERING RITMOS
% =========================================================================
% Explanation:
%   Irradiance values given by DIRSIG is converted to voltage. This
%   radiometric process accounts for the filter effects, dispersive
%   elements, detector elements, shot noise, A/D converters with readout
%   circuitry, integration time.
% Cited Paper : Feature Aided Tracking with Hyperspectral Imagery
% =========================================================================
[rows,cols,bands] = size(data);
sigma = 0.2*10^-17;    % Transmittance of the optical platform
f = 100;        % Focal length of the lens in mm
d = 20.0;       % Aperture diameter in mm 
Ax = 1200 * 17*10^-1;  % Detector array y dimension in cm 
Ay = 800 * 17*10^-1;  % Detector array x dimension in cm
A = Ax * Ay;    % Detector Area
tint = 10^-3;   % Integration time of the sensor in s
V_e = 0.95;     % Charge to voltage conversion (Volts per electron)
QE = ml;       % Quantum efficiency of the sensor (electrons per photon)
hc = 3.16152649*10^-26; % Planck's Constant
b = 5;          % Number of bits used in the processed data
L = 10^4 * data;% Sensor Reaching Radiance
G = (1 + 4 * (f/d)^2)/(sigma*pi);  % Optical Throughput Equation
factor=tint * A * V_e * QE * (0.7*10^-4)^2/(hc);
E = L./G;       % Camera Equation
S = E.*factor;  % Irradiance to Voltage Conversion
Vsat = 10^4;    % Maximum signal level

%% ADD READ, SHOT NOISE AND ESTIMATE FINAL VOLTAGE
Nro = 0 + 30.*randn([rows cols bands]) * V_e;
Nphi = 0 + sqrt(abs(S)).*randn([rows cols bands]);
Sfn = ((2^b.*(max(min(S+Nphi+Nro,Vsat),0))./(Vsat))+0.5).*2^-b; 
data_Sfinal = Sfn; 
