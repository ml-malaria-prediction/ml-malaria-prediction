function layer = sequenceInputLayer(inputSize,nameValueArgs)
% sequenceInputLayer   Sequence input layer
%
%   layer = sequenceInputLayer(inputSize) defines a sequence input layer.
%   inputSize is the size of the input sequence at each time step,
%   specified as a positive integer or vector of positive integers.
%       - For vector sequence input, inputSize is a scalar corresponding to
%       the number of features.
%       - For 1-D image sequence input, inputSize is a vector of two
%       elements [H C], where H is the input height, and C is the number of 
%       channels of the image.
%       - For 2-D image sequence input, inputSize is a vector of three
%       elements [H W C], where H is the image height, W is the image
%       width, and C is the number of channels of the image.
%       - For 3-D image sequence input, inputSize is a vector of four
%       elements [H W D C], where H is the image height, W is the image
%       width, D is the image depth, and C is the number of channels of the
%       image.
%
%   layer = sequenceInputLayer(inputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%    'Normalization'     Data normalization applied when data is forward
%                        propagated through the input layer. Valid values
%                        are:
%                           'zerocenter'        - zero-center normalization
%                           'zscore'            - z-score normalization
%                           'rescale-symmetric' - rescale to [-1 1]
%                           'rescale-zero-one'  - rescale to [0 1]
%                           'none'              - do not normalize
%                           function handle     - custom normalization
%
%                        Default: 'none'
%
%    'NormalizationDimension'
%                        Dimension over which the same normalization is
%                        applied, specified as one of the following:
%                          - 'auto'    - If the ResetInputNormalization
%                                        training option is false and you
%                                        specify normalization statistics,
%                                        then normalize over dimensions
%                                        matching the statistics.
%                                        Otherwise, recompute the
%                                        statistics and apply channel-wise
%                                        normalization.
%                          - 'channel' - Channel-wise normalization
%                          - 'element' - Element-wise normalization
%                          - 'all'     - Normalize all values using
%                                        scalar statistics
%
%                        Default: 'auto'
%
%    'Mean'              The mean used for zero centering and z-score normalization.
%                        This can be [], a scalar, or a numeric array.
%                           - For vector sequence input, 'Mean' can be a scalar, a
%                             inputSize-by-1  vector of means per channel, or [].
%                           - For 1-D image sequence input, 'Mean' can be a scalar, a
%                             1-by-C array of means per channel, an H-by-C
%                             array, or [], where [H C] is the input size.
%                           - For 2-D image sequence input, 'Mean' can be a scalar, a
%                             1-by-1-by-C array of means per channel, a H-by-W-by-C
%                             array, or [], where [H W C] is the input size.
%                           - For 3-D image sequence input, 'Mean' can be a scalar, a
%                             1-by-1-by-1-by-C array of means per channel, a 
%                             H-by-W-by-D-by-C array, or [], where [H W D C] is the 
%                             input size.
%
%                        Default: []
%
%    'StandardDeviation'              
%                        The standard deviation used for z-score normalization.
%                        This can be [], a scalar, or a numeric array.
%                           - For vector sequence input, 'StandardDeviation' 
%                             can be a scalar, a inputSize-by-1  vector of means
%                             per channel, or [].
%                           - For 1-D image sequence input, 'StandardDeviation'
%                             can be a scalar, a 1-by-C array of means per 
%                             channel, an H-by-C array, or [], where [H C] 
%                             is the input size.
%                           - For 2-D image sequence input, 'StandardDeviation' 
%                             can be a scalar, a 1-by-1-by-C array of means 
%                             per channel, a H-by-W-by-C array, or [], where 
%                             [H W C] is the input size.
%                           - For 3-D image sequence input, 'StandardDeviation' 
%                             can be a scalar, a 1-by-1-by-1-by-C array of means  
%                             per channel, a H-by-W-by-D-by-C array, or [],
%                             where [H W D C] is the input size.
%
%                        Default: []
%
%    'Min'               The minimum used for rescaling.
%                        This can be [], a scalar, or a numeric array.
%                           - For vector sequence input, 'Min' can be a  
%                             scalar, a inputSize-by-1  vector of means per
%                             channel, or [].
%                           - For 1-D image sequence input, 'Min' can be a scalar, a
%                             1-by-C array of means per channel, an H-by-C
%                             array, or [], where [H C] is the input size.
%                           - For 2-D image sequence input, 'Min' can be a 
%                             scalar, a 1-by-1-by-C array of means per
%                             channel, a H-by-W-by-C array, or [], where 
%                             [H W C] is the input size.
%                           - For 3-D image sequence input, 'Min' can be a
%                             scalar, a 1-by-1-by-1-by-C array of means  
%                             per channel, a H-by-W-by-D-by-C array, or [],
%                             where [H W D C] is the input size.
%
%                        Default: []
%
%    'Max'               The maximum used for rescaling.
%                        This can be [], a scalar, or a numeric array.
%                           - For vector sequence input, 'Max' can be a  
%                             scalar, a inputSize-by-1  vector of means per
%                             channel, or [].
%                           - For 1-D image sequence input, 'Max' can be a scalar, a
%                             1-by-C array of means per channel, an H-by-C
%                             array, or [], where [H C] is the input size.
%                           - For 2-D image sequence input, 'Max' can be a 
%                             scalar, a 1-by-1-by-C array of means per
%                             channel, a H-by-W-by-C array, or [], where 
%                             [H W C] is the input size.
%                           - For 3-D image sequence input, 'Max' can be a
%                             scalar, a 1-by-1-by-1-by-C array of means  
%                             per channel, a H-by-W-by-D-by-C array, or [],
%                             where [H W D C] is the input size.
%
%                        Default: []
%
%    'MinLength'         Minimum sequence length the input layer accepts, 
%                        specified as a positive integer. When training or 
%                        making predictions with the network, if the input 
%                        data has fewer than MinLength time steps, the 
%                        software throws an error.
%
%                        Default: 1
%
%    'SplitComplexInputs'         
%                        Flag indicating whether the layer should split the 
%                        real and imaginary components of the data, 
%                        specified as one of the following:
%                          - false - Do not split the input data. This 
%                                    option supports real input only. 
%                          - true  - Output real and imaginary parts as
%                                    separate channels. The number of 
%                                    channels of the layer output is 2*C, 
%                                    where C is the number of channels in 
%                                    the input data. For real input, this
%                                    option outputs zeros for the imaginary 
%                                    parts.
%                        When SplitComplexInputs is 1, Normalization must 
%                        be 'zerocenter', 'zscore', 'none', or a function 
%                        handle.
%
%                        Default: false
%
%    'Name'              A name for the layer.
%
%                        Default: ''
%
%   Example:
%       % Create a sequence input layer for multi-dimensional time series
%       % with 5 dimensions per time step.
%
%       layer = sequenceInputLayer(5);
%
%   See also nnet.cnn.layer.SequenceInputLayer
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2017-2023 The MathWorks, Inc.

arguments
    inputSize {mustBeNumeric, mustBeNonempty, mustBeValidScalarOrRowVector(inputSize), ...
        mustBeReal, mustBeFinite, mustBeInteger, mustBePositive}
    nameValueArgs.Normalization = 'none';
    nameValueArgs.NormalizationDimension = 'auto';
    nameValueArgs.Name {iAssertValidLayerName} = '';
    nameValueArgs.Mean = [];
    nameValueArgs.StandardDeviation = [];
    nameValueArgs.Min = [];
    nameValueArgs.Max = [];
    nameValueArgs.MinLength(1,1) {mustBeNumeric, mustBeInteger, mustBePositive} = 1;
    nameValueArgs.AverageImage = [];
    nameValueArgs.SplitComplexInputs (1,1) {iAssertBinary} = false;
end

inputSize = nnet.internal.cnn.layer.util.gatherParametersToCPU(inputSize);
nameValueArgs = nnet.internal.cnn.layer.util.gatherParametersToCPU(nameValueArgs);
[inputSize,nameValueArgs] = iConvertToCanonicalForm(inputSize,nameValueArgs);

normalization = iCreateTransforms(...
    nameValueArgs.Normalization, inputSize);

% Create an internal representation of a sequence input layer.
internalLayer = nnet.internal.cnn.layer.SequenceInput(...
    nameValueArgs.Name, ...
    inputSize, ...
    normalization, ...
    nameValueArgs.MinLength, ...
    nameValueArgs.SplitComplexInputs);

% Assign statistics
internalLayer.Mean = nameValueArgs.Mean;
internalLayer.Std = nameValueArgs.StandardDeviation;
internalLayer.Min = nameValueArgs.Min;
internalLayer.Max = nameValueArgs.Max;
internalLayer.NormalizationDimension = nameValueArgs.NormalizationDimension;

% Pass the internal layer to a function to construct a user visible
% sequence input layer.
layer = SequenceInputLayerClass(internalLayer);
end

function x = iCheckAndReturnValidNormalization(x, splitComplexInputs)
validateattributes(x,{'string','char','function_handle'},{})
if isa(x,'function_handle')
    % Checks are performed at train/inference time when normalization is
    % applied
else
    if splitComplexInputs
        % Rescaling is not supported for complex inputs
        validTransforms = {'zerocenter', 'zscore', 'none'};
        try
            x = validatestring(x, validTransforms);
        catch e
            error(message('nnet_cnn:layer:InputLayer:InvalidNormalizationForComplex'));
        end
    else
        validTransforms = {'zerocenter', 'zscore', 'rescale-symmetric', ...
            'rescale-zero-one', 'none'};
        x = validatestring(x, validTransforms, '', 'Normalization');
    end
end
end

function mustBeValidScalarOrRowVector(sz)
isValidSize = (isscalar(sz) || iIsValidRowVector(sz));
if ~isValidSize
    error(message('nnet_cnn:layer:SequenceInputLayer:InvalidInputSize'));
end
end

function iAssertBinary(value)
nnet.internal.cnn.options.OptionsValidator.assertBinary(value);
end

function tf = iIsValidRowVector(x)
tf = isrow(x) && (numel(x) < 5);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function [inputSize,args] = iConvertToCanonicalForm(inputSize,params)
try    
    args = struct;
    % Make sure integral values are converted to double
    inputSize = iConvertToDouble(inputSize);
    args.Normalization = iCheckAndReturnValidNormalization(params.Normalization, ...
        params.SplitComplexInputs);
    args.NormalizationDimension = iCheckAndReturnValidDimension( ...
        params.NormalizationDimension, args.Normalization );
    args.Mean = iCheckAndReturnSingleStatistics( params.Mean, 'Mean', args.Normalization, ...
        args.NormalizationDimension, inputSize, true, params.SplitComplexInputs);
    args.StandardDeviation = iCheckAndReturnSingleStatistics( params.StandardDeviation, 'StandardDeviation', ...
        args.Normalization, args.NormalizationDimension, inputSize);
    args.Min = iCheckAndReturnSingleStatistics(params.Min, 'Min', ...
        args.Normalization, args.NormalizationDimension, inputSize);
    args.Max = iCheckAndReturnSingleStatistics(params.Max, 'Max', ...
        args.Normalization, args.NormalizationDimension, inputSize);
    args.MinLength = iConvertToDouble(params.MinLength);
    args.SplitComplexInputs = logical(params.SplitComplexInputs);

    % Make sure strings get converted to char vectors
    args.Name = convertStringsToChars(params.Name); 
catch e
    % Reduce the stack trace of the error message by throwing as caller
    throwAsCaller(e)
end
end

function statsValue = iCheckAndReturnSingleStatistics( userValue, name, ...
    normalization, normalizationDimension, inputSize, allowComplexStatistic, ...
    splitComplexInputs)

if nargin < 6
    allowComplexStatistic = false;
end

if nargin < 7
    splitComplexInputs = false;
end

% Validate user-provided statistics with respect to normalization method 
% and input size. Return the statistic in single precision.
if ~isempty(userValue)
    nnet.internal.cnn.layer.paramvalidation.validateNormalizationStatistics(...
        userValue, name, normalization, normalizationDimension, inputSize, ...
        'HasRowVectorStats', false, ...
        'AllowComplexStatistic', allowComplexStatistic, ...
        'SplitComplexInputs', splitComplexInputs);
end
statsValue = single(userValue);
end

function transform = iCreateTransforms(type, dataSize)
transform = nnet.internal.cnn.layer.InputTransformFactory.create(type, dataSize);
end

function dimValue = iCheckAndReturnValidDimension(dimValue,normalization)
dimValue = nnet.internal.cnn.layer.paramvalidation.validateNormalizationDimension(dimValue,normalization);
end

function value = iConvertToDouble(value)
value = nnet.internal.cnn.layer.util.convertToDouble(value);
end
