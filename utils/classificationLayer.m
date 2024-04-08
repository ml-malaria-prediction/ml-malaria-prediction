function layer = classificationLayer( varargin )
% classificationLayer   Classification output layer for a neural network
%
%   layer = classificationLayer() creates a classification output layer for
%   a neural network. The classification output layer holds the name of the
%   loss function that is used for training the network, the size of the
%   output, and the class labels.
%
%   layer = classificationLayer('PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Name'            - A name for the layer. The default is ''.
%
%       'Classes'         - Categories of the output layer, specified as a
%                           string vector, a categorical vector, a cell
%                           array of character vectors, or 'auto'. The
%                           function sets the Classes property of the
%                           output layer. For string or cell array input Y,
%                           the function sets the classes in the order that
%                           they appear in Y. The classes are assumed to be
%                           nonordinal. For categorical input Y, the
%                           function sets the classes in the order
%                           categories(Y). If Y is ordinal, then the
%                           Classes property of the output layer is
%                           ordinal. Otherwise, it is nonordinal. If the
%                           value is 'auto',  the classes are automatically
%                           set during training. Default: 'auto'.
%
%      'ClassWeights'       Weights for weighted cross entropy loss,
%                           specified as a vector of positive weights or
%                           'none'. Each element specifies the weight of
%                           the corresponding class in Classes.
%
%                           To specify 'ClassWeights', you must also 
%                           specify 'Classes'. When specifying
%                           'ClassWeights', 'Classes' must not be 'auto'.
%                           Default: 'none'.
%
%   Example:
%       % Create a classification output layer.
%
%       layer = classificationLayer();
%
%   See also nnet.cnn.layer.ClassificationOutputLayer, softmaxLayer.
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2015-2020 The MathWorks, Inc.

% Parse the input arguments
args = ClassificationOutputLayer.parseInputArguments(varargin{:});

% Create an internal representation of a cross entropy layer
internalLayer = nnet.internal.cnn.layer.CrossEntropy( ...
    args.Name,...
    args.NumClasses,...
    args.Classes,...
    args.ClassWeights,...
    4); 

% Pass the internal layer to a function to construct
layer = ClassificationOutputLayer(internalLayer);

end