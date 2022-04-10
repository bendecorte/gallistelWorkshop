function Dkl = normDkl(mu1,mu2,sig1,sig2)
% KL divergence in nats; to convert to bits, multiply by 1.44
Dkl = log(sig2/sig1) + (sig1^2+(mu1-mu2)^2)/(2*sig2^2) -1/2;