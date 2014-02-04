% Use a EigenGPARD model to predict 
% Parameters:
% model - trained model
% testX - test data
%    N by D matrix. Each row is a data point.

function [mu s2] = EigenGPARD_pred(model, X, t, testX)
% Load model
sigma2 = exp(model.logSigma*2);
eta = exp(model.logEta);
a0 = exp(model.logA0);
B = model.B;
% to avoid semi positive definite
epsilon = 1e-10;
% for later use
X2 = X.*X;
B2 = B.*B;
X_eta = bsxfun(@times,X,eta');
B_eta = bsxfun(@times,B,eta');
% Compute gram matrices
expH = exp(bsxfun(@minus,bsxfun(@minus,2*X_eta*B',X2*eta),(B2*eta)'));
Kxb = a0*expH;
expF = exp(bsxfun(@minus,bsxfun(@minus,2*B_eta*B',B2*eta),(B2*eta)'));
Kbb = a0*expF+epsilon*eye(size(B,1));
% Define Q = Kbb + 1/sigma2 * Kbx *Kxb
Q = Kbb+(Kxb'*Kxb)/sigma2;
% Cholesky factorization for stable computation
cholQ = chol(Q,'lower');
lowerOpt.LT = true; upperOpt.LT = true; upperOpt.TRANSA = true;
invCholQ_Kbx_invSigma2 = linsolve(cholQ,Kxb'/sigma2,lowerOpt);
invKbb_Kbx_invCN = linsolve(cholQ,invCholQ_Kbx_invSigma2,upperOpt);
invKbb_Kbx_invCN_t = invKbb_Kbx_invCN*t;
% number of test data points
Ns = size(testX,1);
% number of data points per mini batch
nperbatch = 1000;
% number of already processed test data points
nact = 0;
% Allocate memory for predictive mean
mu = zeros(Ns,1);
% Allocate memory for predictive variance
s2 = mu;

% process minibatches of test cases to save memory
while nact<Ns
    % Data points to process
    id = (nact+1):min(nact+nperbatch,Ns);
    % Cross Kernel Matrix between the testing points and basis points
    Ksb = a0*exp(bsxfun(@minus,bsxfun(@minus,2*testX(id,:)*B_eta',testX(id,:).*testX(id,:)*eta),(B2*eta)'));
    % Predictive mean
    mu(id) = Ksb * invKbb_Kbx_invCN_t;
    % Predictive variance
    s2(id) = sum(linsolve(cholQ,linsolve(cholQ,Ksb',lowerOpt),upperOpt)'.*Ksb,2)+sigma2+epsilon;
    nact = id(end);
end
end