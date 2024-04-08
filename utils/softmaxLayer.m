function layer = softmaxLayer( nameValueArgs )
% softmaxLayer   Softmax layer
%
%   layer = softmaxLayer() creates a softmax layer. This layer is
%   useful for classification problems.
%
%   layer = softmaxLayer('PARAM1', VAL1) specifies optional parameter
%   name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example:
%       % Create a softmax layer.
%
%       layer = softmaxLayer();
%
%   See also nnet.cnn.layer.SoftmaxLayer, classificationLayer,
%   fullyConnectedLayer.
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2015-2021 The MathWorks, Inc.

arguments
    nameValueArgs.Name {iAssertValidLayerName} = '';
end
nameValueArgs.Name = convertStringsToChars(nameValueArgs.Name);

% Create an internal representation of a softmax layer.
internalLayer = nnet.internal.cnn.layer.Softmax(nameValueArgs.Name);

% Pass the internal layer to a function to construct a user visible
% softmax layer.
layer = SoftmaxLayerClass(internalLayer);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end
