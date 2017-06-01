% How to plot a simple gaussian process
% credits:
%   https://www.cs.ubc.ca/~hutter/EARG.shtml/earg/papers05/rasmussen_gps_in_ml.pdf


xs = (-5:0.2:5)';
ns = size(xs, 1);
keps = 1e-9;
m = @(x) 0.25*x.^2;
K = @(p, q) exp(-0.5*(repmat(p', size(q)) - repmat(q,size(p'))).^2);
fs = m(xs) + chol(K(xs, xs)+keps*eye(ns))'*randn(ns, 1);
% fs = m(xs) + chol(K(xs, xs)+keps*eye(ns))'*ones(ns,1);
plot(xs, fs, '-');