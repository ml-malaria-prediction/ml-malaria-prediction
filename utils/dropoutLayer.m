function layer = dropoutLayer(probability, NameValueArgs)
% dropoutLayer   Dropout layer
%
%   layer = dropoutLayer creates a dropout layer that randomly sets
%   input elements to zero with a probability of 0.5 during training.
%   During prediction, the layer has no effect. This can help prevent 
%   overfitting.
%
%   layer = dropoutLayer(probability) creates a dropout layer with
%   dropout probability specified by a nonnegative number less than 1.
%
%   layer = dropoutLayer(__, 'Name',name) optionally specifies a name 
%   for the layer in addition to using any of the previous syntax. The 
%   default name is ''.
%
%   Example:
%       % Create a dropout layer with dropout probability 0.4.
%
%       layer = dropoutLayer(0.4);
%
%   See also nnet.cnn.layer.DropoutLayer, imageInputLayer, reluLayer.
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2015-2023 The MathWorks, Inc.

arguments
    probability         {iAssertValidProbability} = 0.5
    NameValueArgs.Name  {iAssertValidLayerName} = ''
end

% Store all input arguments in one struct
args = NameValueArgs;
args.Probability = probability;

% Gather arguments to CPU and convert them to canonical form
args = nnet.internal.cnn.layer.util.gatherParametersToCPU(args);
args = iConvertToCanonicalForm(args);

% Create an internal representation of a dropout layer.
internalLayer = nnet.internal.cnn.layer.Dropout( ...
    args.Name, ...
    args.Probability);

% Pass the internal layer to a  function to construct a user visible
% dropout layer.
layer = DropoutLayerClass(internalLayer);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function iAssertValidProbability(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','real','finite','>=',0,'<',1});
end

function inputArguments = iConvertToCanonicalForm(params)
inputArguments = struct;
inputArguments.Probability = iConvertToDouble(params.Probability);
inputArguments.Name = char(params.Name); % make sure strings get converted to char vectors
end

function value = iConvertToDouble(value)
value = nnet.internal.cnn.layer.util.convertToDouble(value);
end
