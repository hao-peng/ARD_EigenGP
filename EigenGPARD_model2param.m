function param = EigenGPARD_model2param(model, D, M)
param = [model.logSigma; model.logEta; model.logA0; reshape(model.B, D*M, 1)];
end