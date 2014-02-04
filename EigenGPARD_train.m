% Train EigenGPARD model
% Parameters:
% initModel - initial values for parameters of EigenGPARD
%    initModel requires logSigma, logEta, logA0, logA1 and logA2
%    if B is not initialized in initModel, we use kmeans+ to initialize it.
% trainX - training data
%    N by D matrix. Each row is a data point.
% trainY - tarining labels
%    N by 1 matrix. Each row is a label.
% M - number of basis used
% nIter - maximum number of iterations for optimization

function model = EigenGPARD_train(initModel, trainX, trainY, M, nIter)
% Get the dimension of input
D = size(trainX, 2);

% Initialize B if it is not given
if ~isfield(initModel, 'B')
    %use kmeans with both x and y
    [IDX, B] = fkmeans(trainX', M);
    initModel.B = B;
    clear IDX;
end
param = EigenGPARD_model2param(initModel, D, M);

% Train EigenGPARD by minimizing the log likelihood.
[new_param] = minimize(param, @(param) EigenGPARD_negLogLik(param, trainX, trainY, M), nIter);
model = EigenGPARD_param2model(new_param, D, M);

end