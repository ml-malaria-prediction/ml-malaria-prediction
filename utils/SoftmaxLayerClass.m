classdef SoftmaxLayerClass < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % SoftmaxLayer   Softmax layer
    %
    %   To create a softmax layer, use softmaxLayer.
    %   This layer is useful for classification problems.
    %
    %   SoftmaxLayer properties:
    %       Name                   - A name for the layer
    %       NumInputs              - The number of inputs of the layer.
    %       InputNames             - The names of the inputs of the layer.
    %       NumOutputs             - The number of outputs of the layer.
    %       OutputNames            - The names of the outputs of the layer.
    %
    %   Example:
    %       Create a softmax layer.
    %
    %       layer = softmaxLayer();
    %
    %   See also softmaxLayer
    
    %   Copyright 2015-2019 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    methods
        function this = SoftmaxLayerClass(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            out.Version = 4.0;
            out.Name = this.PrivateLayer.Name;
            out.ChannelDim = this.PrivateLayer.ChannelDim;
            out = iAddPropertiesForForwardCompatibility(out);
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
    end
    
    methods(Hidden, Static)
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            if in.Version <= 2
                in = iUpgradeVersionTwoToVersionThree(in);
            end
            if in.Version <= 3
                in = iUpgradeVersionThreeToVersionFour(in);
            end
            this = iLoadSoftmaxLayerFromCurrentVersion(in);
        end
    end
    
    methods(Access = protected)
        function [description, type] = getOneLineDisplay(~)
            description = iGetMessageString('nnet_cnn:layer:SoftmaxLayer:oneLineDisplay');
            
            type = iGetMessageString( 'nnet_cnn:layer:SoftmaxLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            groups = this.propertyGroupGeneral( {'Name'} );
        end
    end
end

function messageString = iGetMessageString( varargin )
    messageString = getString( message( varargin{:} ) );
end

function out = iAddPropertiesForForwardCompatibility(out)

% In 2017b-2019b, we can only load layers if they have a "VectorFormat"
% property. This property will be true if "ChannelDim" in 2020a has a value
% of 1.
if out.ChannelDim == 1
    out.VectorFormat = true;
else
    out.VectorFormat = false;
end

end

function S = iUpgradeVersionOneToVersionTwo(S)
    % iUpgradeVersionOneToVersionTwo   Upgrade a v1 (2016a-2017a) saved struct
    % to a v2 saved struct. This means adding a "VectorFormat" property.
    
    S.Version = 2;
    S.VectorFormat = false;
end

function S = iUpgradeVersionTwoToVersionThree(S)
    % iUpgradeVersionTwoToVersionThree   Upgrade a v2 (2017b-2018b) saved struct
    % to a v3 saved struct. This means adding a "ChannelDim" property.
    
    S.Version = 3;
    % ChannelDim is used for non-vector formats.
    % In releases prior to 19a, only 4-D data was supported.
    if S.VectorFormat == false
        S.ChannelDim = 3;
    else
        S.ChannelDim = [];
    end
end

function S = iUpgradeVersionThreeToVersionFour(S)
    % iUpgradeVersionTwoToVersionThree   Upgrade a v3 (2019a-2019b) saved struct
    % to a v4 saved struct. This means ignoring the "VectorFormat" property and
    % replacing empty "ChannelDim" with 1.
    
    S.Version = 4;
    if S.VectorFormat == true || isempty(S.ChannelDim)
        S.ChannelDim = 1;
    end
end

function layer = iLoadSoftmaxLayerFromCurrentVersion(in)
internalLayer = nnet.internal.cnn.layer.Softmax(in.Name);
internalLayer.ChannelDim = in.ChannelDim;
internalLayer = internalLayer.setupForHostPrediction();
layer = nnet.cnn.layer.SoftmaxLayer(internalLayer);
end

function iAssertValidLayerName(name)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLayerName(name));
end

function iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    func();
catch exception
    throwAsCaller(exception)
end
end