function d = expDKL(Lt,La)
% Syntax     d = expDKL(Lt,La)
% gives Kullback-Leibler divergence of La (approximating distribution) from
% Lt (true distribution) in nats (= 1.44 bits), where Lt and La are vectors
% or arrays of rate values
d = log(Lt) - log(La) + La./Lt -1;