function [ indx ] = resampleStratified(w, N)
% [ indx ] = resampleStratified(w, N)
% Stratified resampling method for particle filtering. Author: T. Li, Ref:
% T. Li, M. Bolic, P. Djuric, Resampling methods for particle filtering, 
% submit to IEEE Signal Processing Magazine, August 2013

% Input:
%       w    the input weight sequence 
%       N    the desired length of the output sequence(i.e. the desired number of resampled particles)
% Output:
%       indx the resampled index according to the weight sequence

if nargin == 1
  N = length(w);
end
M = length(w);
w = w / sum(w);
Q = cumsum(w);
indx = zeros(1, N);
T = linspace(0, 1-1/N, N) + rand(1, N)/N;

i = 1;
j = 1;
while(i<= N && j<= M),
    while Q(j) < T(i)
        j = j + 1;
    end
    indx(i) = j;
    i = i + 1;
end

