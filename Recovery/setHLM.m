function setHLM(runModelNum,whichJAGS,whichQuals,doParallel)
%% setHLM
% setHLM sets up multiple HLM models to run sequentially according to inputs
% This function takes the following inputs:
% runModelNum - 1=parameter estimation of eta,
%               2=model selection between 3 models with parameter expansion of model indicator, 
%               3=model selection between two models with parameter expansion of model indicator,
%               4=parameter retrival for PT_original, 
%               5=model selection between 4 models with parameter expansion of model indicator,
%               6=parameter retrival for 1 model. 
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs 
%               to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% 
% There are three qualities for several variables, each selected by whichQuals
% qualities  are 'bronze','silver','gold'
% gold is highest quality but takes longest, bronzest lowest but fastest
% etc.

%% Specifies qualities to be selected from
numRuns      = length(whichQuals);%how many separate instances of an MCMC to run
nBurnin      = [1e2,1e3,1e4,2e4,4e4];%from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];%from 50 to 20k
nChains      = [4,4,4,4,4];%
nThin        = 1;%thinnning factor, 1 = no thinning, 2=every 2nd etc.
%% Runs HLMs sequentiallt
for i=1:numRuns
    computeHLM(runModelNum,nBurnin,nSamples,nThin,nChains,subjList,whichJAGS,runPlots,synthMode,doParallel)
end