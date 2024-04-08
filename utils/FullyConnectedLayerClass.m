classdef FullyConnectedLayerClass < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % FullyConnectedLayer   Fully connected layer
    %
    %   To create a fully connected layer, use fullyConnectedLayer
    %
    %   A fully connected layer. This layer has weight and bias parameters
    %   that are learned during training.
    %
    %   FullyConnectedLayer properties:
    %       Name                        - A name for the layer.
    %       InputSize                   - The input size of the fully
    %                                     connected layer.
    %       OutputSize                  - The output size of the fully
    %                                     connected layer.
    %       Weights                     - The weight matrix.
    %       Bias                        - The bias vector.
    %       WeightsInitializer          - The function for initializing the 
    %                                     weights.
    %       WeightLearnRateFactor       - The learning rate factor for the
    %                                     weights.
    %       WeightL2Factor              - The L2 regularization factor for
    %                                     the weights.
    %       BiasInitializer             - The function for initializing the
    %                                     bias.
    %       BiasLearnRateFactor         - The learning rate factor for the
    %                                     bias.
    %       BiasL2Factor                - The L2 regularization factor for
    %                                     the bias.
    %       NumInputs                   - The number of inputs for the 
    %                                     layer.
    %       InputNames                  - The names of the inputs of the 
    %                                     layer.
    %       NumOutputs                  - The number of outputs of the 
    %                                     layer.
    %       OutputNames                 - The names of the outputs of the 
    %                                     layer.
    %
    %   Example:
    %       Create a fully connected layer with an output size of 10, and an
    %       input size that will be determined at training time.
    %
    %       layer = fullyConnectedLayer(10);
    %
    %   See also fullyConnectedLayer
    
    %   Copyright 2015-2022 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        % InputSize   The input size for the layer
        %   The input size for the fully connected layer. If this is set to
        %   'auto', then the input size will be automatically set at
        %   training time.
        InputSize
        
        % OutputSize   The output size for the layer
        %   The output size for the fully connected layer.
        OutputSize       
    end
    
    properties(Dependent)
        % Weights   The weights for the layer
        %   The weight matrix for the fully connected layer. This matrix
        %   will have size OutputSize-by-InputSize.
        Weights
        
        % Bias   The biases for the layer
        %   The bias vector for the fully connected layer. This vector will
        %   have size OutputSize-by-1.
        Bias
        
        % WeightsInitializer   The function to initialize the weights.
        WeightsInitializer
        
        % WeightLearnRateFactor   The learning rate factor for the weights
        %   The learning rate factor for the weights. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the weights in this layer. For example, if it
        %   is set to 2, then the learning rate for the weights in this
        %   layer will be twice the current global learning rate.
        WeightLearnRateFactor
        
        % WeightL2Factor   The L2 regularization factor for the weights
        %   The L2 regularization factor for the weights. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the weights in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the weights in this layer will be twice the
        %   global L2 regularization setting.
        WeightL2Factor
               
        % BiasInitializer   The function to initialize the bias.
        BiasInitializer
        
        % BiasLearnRateFactor   The learning rate factor for the biases
        %   The learning rate factor for the bias. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the bias in this layer. For example, if it
        %   is set to 2, then the learning rate for the bias in this layer
        %   will be twice the current global learning rate.
        BiasLearnRateFactor
        
        % BiasL2Factor   The L2 regularization factor for the biases
        %   The L2 regularization factor for the biases. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the biases in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the biases in this layer will be twice the
        %   global L2 regularization setting.
        BiasL2Factor
    end
    
    methods
        function this = FullyConnectedLayerClass(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            % The observationDim is not saved as it is inferred from the
            % InputSize during the loadobj stage.
            privateLayer = this.PrivateLayer;
            out.Version = 3.0;
            out.Name = privateLayer.Name;
            out.InputSize = privateLayer.InputSize;
            out.OutputSize = iGetOutSize(privateLayer);
            out.Weights = toStruct(privateLayer.Weights);
            out.Bias = toStruct(privateLayer.Bias);
        end
        
        function val = get.InputSize(this)
            if ~isempty(this.PrivateLayer.InputSize)
                % Get the input size from the internal 4-D input size.
                val = prod(this.PrivateLayer.InputSize);
            elseif ~isempty(this.Weights)
                val = size(this.Weights, 2);
            else
                val = 'auto';
            end
        end
        
        function val = get.OutputSize(this)
            val = this.PrivateLayer.NumNeurons;
        end
        
        function weights = get.Weights(this)
            privateWeights = this.PrivateLayer.Weights.HostValue;
            if isa(privateWeights, 'dlarray')
                privateWeights = extractdata(privateWeights);
            end
            weights = nnet.internal.cnn.layer.util.FullyConnectedWeightsConverter.toExternal(...
                privateWeights,this.OutputSize);
        end
        
        function this = set.Weights(this, value)
            expectedSize = this.PrivateLayer.ExtWeightsSize;
            attributes = {'size', expectedSize, ...
                'real', 'nonsparse'};
            value = iGatherAndValidateParameter(value, attributes);

            this.PrivateLayer.Weights.Value = value;
        end
        
        function val = get.Bias(this)
            val = this.PrivateLayer.Bias.HostValue;
            if(~isempty(val))
                if isa(val, 'dlarray')
                    val = extractdata(val);
                end        
                val = nnet.internal.cnn.layer.util.FullyConnectedBiasConverter.toExternal(...
                    val, this.OutputSize);
            end
        end
        
        function this = set.Bias(this, value)
            attributes = {'column', 'real', 'nonsparse', 'nrows', ...
                this.OutputSize};
            value = iGatherAndValidateParameter(value, attributes);          
            this.PrivateLayer.Bias.Value = value;
        end
        
        function val = get.WeightsInitializer(this)
            if iIsCustomInitializer(this.PrivateLayer.Weights.Initializer)
                val = this.PrivateLayer.Weights.Initializer.Fcn;
            else
                val = this.PrivateLayer.Weights.Initializer.Name;
            end
        end
        
        function this = set.WeightsInitializer(this, value)
            value = iAssertValidWeightsInitializer(value);        
            % Create the initializer with in and out indices of the weights
            % size: OutputSize-by-InputSize
            this.PrivateLayer.Weights.Initializer = ...
                iInitializerFactory(value, 2, 1);
        end
        
        function val = get.WeightLearnRateFactor(this)
            val = this.PrivateLayer.Weights.LearnRateFactor;
        end
        
        function this = set.WeightLearnRateFactor(this, value)
            value = gather(value);
            iAssertValidFactor(value,'WeightLearnRateFactor');
            this.PrivateLayer.Weights.LearnRateFactor = value;
        end
        
        function val = get.WeightL2Factor(this)
            val = this.PrivateLayer.Weights.L2Factor;
        end
        
        function this = set.WeightL2Factor(this, value)
            value = gather(value);
            iAssertValidFactor(value,'WeightL2Factor');
            this.PrivateLayer.Weights.L2Factor = value;
        end
        
        function val = get.BiasInitializer(this)
            if iIsCustomInitializer(this.PrivateLayer.Bias.Initializer)
                val = this.PrivateLayer.Bias.Initializer.Fcn;
            else
                val = this.PrivateLayer.Bias.Initializer.Name;
            end
        end
        
        function this = set.BiasInitializer(this, value)
            value = iAssertValidBiasInitializer(value);
            % Bias initializers do not require to pass in and out indices
            this.PrivateLayer.Bias.Initializer = iInitializerFactory(value);
        end
        
        function val = get.BiasLearnRateFactor(this)
            val = this.PrivateLayer.Bias.LearnRateFactor;
        end
        
        function this = set.BiasLearnRateFactor(this, value)
            value = gather(value);
            iAssertValidFactor(value,'BiasLearnRateFactor');
            this.PrivateLayer.Bias.LearnRateFactor = value;
        end
        
        function val = get.BiasL2Factor(this)
            val = this.PrivateLayer.Bias.L2Factor;
        end
        
        function this = set.BiasL2Factor(this, value)
            value = gather(value);
            iAssertValidFactor(value,'BiasL2Factor');
            this.PrivateLayer.Bias.L2Factor = value;
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
            this = iLoadFullyConnectedLayerFromCurrentVersion(in);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            outputSizeString = iSizeToStringWithoutDimensionSpecifier( this.OutputSize );
            
            description = iGetMessageString(  ...
                'nnet_cnn:layer:FullyConnectedLayer:oneLineDisplay', ...
                outputSizeString );
            
            type = iGetMessageString( 'nnet_cnn:layer:FullyConnectedLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'InputSize'
                'OutputSize'
                };
            
            learnableParameters = {'Weights', 'Bias'};
            
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( hyperparameters )
                this.propertyGroupLearnableParameters( learnableParameters )
                ];
        end
        
        function footer = getFooter( this )
            variableName = inputname(1);
            footer = this.createShowAllPropertiesFooter( variableName );
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function S = iUpgradeVersionOneToVersionTwo(S)
% iUpgradeVersionOneToVersionTwo   Upgrade a v1 (2016a) saved struct to a v2 saved struct
%   This means gathering the bias and weights from the GPU and putting them
%   on the host.

S.Version = 2;
% As of R2019b, gather should work with or without a GPU.
S.Bias.Value = gather(S.Bias.Value);
S.Weights.Value = gather(S.Weights.Value);
end

function S = iUpgradeVersionTwoToVersionThree(S)
% Add weights and bias initializers set to 'narrow-normal', 'zeros'
S.Version = 3;
S.Weights = iAddBasicInitializerToLearnable(S.Weights, "Normal");
S.Bias = iAddBasicInitializerToLearnable(S.Bias, "Zeros");
end

function s = iAddBasicInitializerToLearnable(s, name)
s.Initializer = struct('Class', ...
    "nnet.internal.cnn.layer.learnable.initializer."+name, ...
    'ConstructorArguments', []);
end

function obj = iLoadFullyConnectedLayerFromCurrentVersion(in)
if ~isempty(in.OutputSize)
    % Remove the singleton dimensions of the Outputsize to construct the
    % internal layer. There can be none, two or three singleton dimensions
    % given an input of sequences, 2D images or 3D data respectively.
    in.OutputSize = in.OutputSize(end);
end
% Set input size to empty so that sizes will be inferred again for the
% right environment (e.g. dlnetwork vs. DAGNetwork)
internalLayer = nnet.internal.cnn.layer.FullyConnected( ...
    in.Name, [], in.OutputSize);
internalLayer.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Weights);
internalLayer.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Bias);

obj = nnet.cnn.layer.FullyConnectedLayer(internalLayer);
end

function iAssertValidFactor(value,factorName)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value,factorName));
end

function iAssertValidLayerName(name)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLayerName(name));
end

function value = iAssertValidWeightsInitializer(value)
name = 'WeightsInitializer';
value = iEvalAndThrow(@() ...
    nnet.internal.cnn.layer.paramvalidation.validateWeightsInitializer(value, name));
end

function value = iAssertValidBiasInitializer(value)
value = iEvalAndThrow(@() ...
    nnet.internal.cnn.layer.paramvalidation.validateBiasInitializer(value));
end

function initializer = iInitializerFactory(varargin)
initializer = nnet.internal.cnn.layer.learnable.initializer.initializerFactory(varargin{:});
end

function tf = iIsCustomInitializer(init)
tf = nnet.internal.cnn.layer.learnable.initializer.util.isCustomInitializer(init);
end

function varargout = iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    [varargout{1:nargout}] = func();
catch exception
    throwAsCaller(exception)
end
end

function outSize = iGetOutSize(privateLayer)
outSize = privateLayer.NumNeurons;
if ~isempty(privateLayer.InputSize)
    outSize = ones(1,numel(privateLayer.InputSize));
    outSize(end) = privateLayer.NumNeurons;
end
end

function value = iGatherAndValidateParameter(varargin)
try
    value = nnet.internal.cnn.layer.paramvalidation...
        .gatherAndValidateNumericParameter(varargin{:});
catch exception
    throwAsCaller(exception)
end
end

function compactSizeString = iSizeToStringWithoutDimensionSpecifier( sizeVector )
% Convert numeric vector to string using the compact floating point conversion character
compactSizeString = num2str(sizeVector, "  %g");
end
