%Author: Dominic Rüfenacht, September 2012
%Purpose: Computes tonemapping function with slope a, and centered at b.

function out = f(in, a, b)
out = 1.0 ./ (1.0 + exp(-a .*((1-in)-b)));