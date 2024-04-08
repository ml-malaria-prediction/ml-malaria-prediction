function layer = fullyConnectedLayer( outputSize, NameValueArgs )
% fullyConnectedLayer   Fully connected layer
%
%   layer = fullyConnectedLayer(outputSize) creates a fully connected
%   layer. outputSize specifies the size of the output for the layer. A
%   fully connected layer will multiply the input by a matrix and then add
%   a bias vector.
%
%   layer = fullyConnectedLayer(outputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Weights'                 - Layer weights, specified as an 
%                                   outputSize-by-inputSize matrix or [],
%                                   where inputSize is the input size of
%                                   the layer. The default is [].
%       'Bias'                    - Layer biases, specified as an
%                                   outputSize-by-1 matrix or []. The
%                                   default is [].
%       'WeightLearnRateFactor'   - A number that specifies multiplier for
%                                   the learning rate of the weights. The
%                                   default is 1.
%       'BiasLearnRateFactor'     - A number that specifies a multiplier
%                                   for the learning rate for the biases.
%                                   The default is 1.
%       'WeightL2Factor'          - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   weights. The default is 1.
%       'BiasL2Factor'            - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   biases. The default is 0.
%       'WeightsInitializer'      - The function to initialize the weights,
%                                   specified as 'glorot', 'he',
%                                   'orthogonal', 'narrow-normal', 'zeros',
%                                   'ones' or a function handle. The
%                                   default is 'glorot'.
%       'BiasInitializer'         - The function to initialize the bias,
%                                   specified as 'narrow-normal', 'zeros', 'ones' 
%                                   or a function handle. The default is 
%                                   'zeros'.
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%   Example 1:
%       % Create a fully connected layer with an output size of 10, and an
%       % input size that will be determined at training time.
%
%       layer = fullyConnectedLayer(10);
%
%   See also nnet.cnn.layer.FullyConnectedLayer, convolution2dLayer,
%   reluLayer.
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2015-2023 The MathWorks, Inc.

arguments
    outputSize                           {iAssertValidOutputSize}
    NameValueArgs.Name                   {iAssertValidLayerName} = ''
    NameValueArgs.WeightLearnRateFactor  {iAssertValidFactor} = 1
    NameValueArgs.BiasLearnRateFactor    {iAssertValidFactor} = 1
    NameValueArgs.WeightsInitializer     = 'glorot'
    NameValueArgs.BiasInitializer        = 'zeros'
    NameValueArgs.WeightL2Factor         {iAssertValidFactor} = 1
    NameValueArgs.BiasL2Factor           {iAssertValidFactor} = 0
    NameValueArgs.Weights                = []
    NameValueArgs.Bias                   = []
end

% Store all input arguments in one struct
args = NameValueArgs;
args.OutputSize = outputSize;

% Gather arguments to CPU and convert them to canonical form
args = nnet.internal.cnn.layer.util.gatherParametersToCPU(args);
args = iConvertToCanonicalForm(args);

% Create an internal representation of a fully connected layer.
internalLayer = nnet.internal.cnn.layer.FullyConnected( ...
    args.Name, ...
    args.InputSize, ...
    args.OutputSize);

internalLayer.Weights.L2Factor = args.WeightL2Factor;
internalLayer.Weights.LearnRateFactor = args.WeightLearnRateFactor;

internalLayer.Bias.L2Factor = args.BiasL2Factor;
internalLayer.Bias.LearnRateFactor = args.BiasLearnRateFactor;

% Pass the internal layer to a function to construct a user visible
% fully connected layer.
layer = FullyConnectedLayerClass(internalLayer);
layer.WeightsInitializer = args.WeightsInitializer;
layer.BiasInitializer = args.BiasInitializer;
layer.Weights = args.Weights;
layer.Bias = args.Bias;
end

function iAssertValidFactor(value)
nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function inputArguments = iConvertToCanonicalForm(params)
% Make sure integral values are converted to double and strings to char vectors
inputArguments = struct;
inputArguments.InputSize = [];
inputArguments.OutputSize = iConvertToDouble( params.OutputSize );
inputArguments.WeightLearnRateFactor = params.WeightLearnRateFactor;
inputArguments.BiasLearnRateFactor = params.BiasLearnRateFactor;
inputArguments.WeightL2Factor = params.WeightL2Factor;
inputArguments.BiasL2Factor = params.BiasL2Factor;
inputArguments.WeightsInitializer = params.WeightsInitializer;
inputArguments.BiasInitializer = params.BiasInitializer;
inputArguments.Name = char(params.Name);
inputArguments.Weights = params.Weights;
inputArguments.Bias = params.Bias;
end

function iAssertValidOutputSize(value)
validateattributes(value, {'numeric'}, ...
    {'nonempty', 'scalar', 'integer', 'positive'});
end

function value = iConvertToDouble(value)
value = nnet.internal.cnn.layer.util.convertToDouble(value);
end
