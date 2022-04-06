function [pp,PP] = berndklexact2(n,k,p)
% Computes the probability, pp, of the nDkl for k successes in n draws from
%  a Bernoulli distribution with parameter theta = p. PP is a 2-col array
% giving the probability of the nDkl for k or fewer and the probability of
% the nDkl for k or more. p can be col vector
%
% Syntax   [pp,PP] = berndklexact(n,k,p,alpha)
%
% Formulae from Peter Latham:
%{
Here it is,
  p(nDkl(k/n||p) > z) = sum_(k=0)^n p(k|n,p) I(nDkl(k/n||p) > z)
where
  p(k|n,p) = p^k (1-p)^(n-k) n!/(k!(n-k)!)
and I is the indicator function: it's 1 if its argument is true and 0
otherwise.
%}
nn = length(p);
if nn>1
    n = repmat(n,nn,1); % making n a col vector same length as p
    k = repmat(k,nn,1); % ditto for k
end
n = n+0; % converts to double (in case n came in as logical)
k = k+0; % ditto for k
pp = p.^k.*(1-p).^(n-k).*factorial(n)./(factorial(k).*factorial(n-k));
% p(k|n,p)

kk = k;
Pgtoreq  = pp;
while kk<n
    kk = kk+1;
    Pgtoreq = Pgtoreq + p.^kk.*(1-p).^(n-kk).*factorial(n)./(factorial(kk).*factorial(n-kk));
end

kk = k;
Pltoreq = pp;
while kk>0
    kk = kk-1;
    Pltoreq = Pltoreq + p.^kk.*(1-p).^(n-kk).*factorial(n)./(factorial(kk).*factorial(n-kk));
end

PP = [Pltoreq Pgtoreq]; % probabilities of nDkl for deviations from p equal
% to or less than the one for k successes and of nDkl for deviations from p
% equal to or greater than k