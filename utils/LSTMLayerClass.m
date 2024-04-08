classdef LSTMLayerClass < nnet.cnn.layer.BaseLSTMLayer
    %LSTMLayer  Long Short-Term Memory (LSTM) layer
    %
    % To create an LSTM layer, use lstmLayer.
    %
    % LSTMLayer properties:
    %     Name                                - Name of the layer.
    %     InputSize                           - Input size of the layer.
    %     NumHiddenUnits                      - Number of hidden units in the layer.
    %     OutputMode                          - Output as sequence or last.
    %     StateActivationFunction             - Activation function to
    %                                           update the cell and hidden
    %                                           state.
    %     GateActivationFunction              - Activation function to
    %                                           apply to the gates.
    %     HasStateInputs                      - Flag indicating whether the 
    %                                           layer has extra inputs
    %                                           that represent the layer
    %                                           states. Specified as one of
    %                                           the following:
    %                                           - false - The layer has
    %                                             a single input with the
    %                                             name 'in'.
    %                                           - true - The layer has
    %                                             two additional inputs
    %                                             with the names 'hidden'
    %                                             and 'cell', which
    %                                             correspond to the hidden
    %                                             state and cell state,
    %                                             respectively.
    %                                           The default is false. When
    %                                           the layer has state inputs,
    %                                           HiddenState and CellState
    %                                           must not be specified.
    %     HasStateOutputs                     - Flag indicating whether the
    %                                           layer should have extra
    %                                           outputs that represent the
    %                                           layer states. Specified as
    %                                           one of the following:
    %                                           - false - The layer has
    %                                             a single output with the
    %                                             name 'out'.
    %                                           - true - The layer has
    %                                             two additional outputs
    %                                             with the names 'hidden'
    %                                             and 'cell', which
    %                                             correspond to the updated
    %                                             hidden state and cell
    %                                             state, respectively.
    %                                           The default is false.
    %     NumInputs                           - Number of inputs for the layer.
    %     InputNames                          - Names of the inputs of the layer.
    %     NumOutputs                          - Number of outputs of the layer.
    %     OutputNames                         - Names of the outputs of the layer.
    %
    % Properties for learnable parameters:
    %     InputWeights                        - Input weights.
    %     InputWeightsInitializer             - The function for
    %                                           initializing the input 
    %                                           weights.    
    %     InputWeightsLearnRateFactor         - Learning rate multiplier
    %                                           for the input weights.
    %     InputWeightsL2Factor                - L2 multiplier for the
    %                                           input weights.
    %
    %     RecurrentWeights                    - Recurrent weights.
    %     RecurrentWeightsInitializer         - The function for
    %                                           initializing the recurrent
    %                                           weights.
    %     RecurrentWeightsLearnRateFactor     - Learning rate multiplier
    %                                           for the recurrent weights.
    %     RecurrentWeightsL2Factor            - L2 multiplier for the
    %                                           recurrent weights.
    %
    %     Bias                                - Bias vector.
    %     BiasInitializer                     - The function for
    %                                           initializing the bias.
    %     BiasLearnRateFactor                 - Learning rate multiplier
    %                                           for the bias.
    %     BiasL2Factor                        - L2 multiplier for the bias.
    %                                       
    % State parameters:
    %     HiddenState                         - Hidden state vector.
    %     CellState                           - Cell state vector.
    %
    %   Example:
    %       Create a Long Short-Term Memory layer.
    %
    %       layer = lstmLayer(10)
    %
    %   See also lstmLayer
    
    %   Copyright 2017-2023 The MathWorks, Inc.
    
    properties(Dependent)
        % InputWeights   The input weights for the layer
        %   The input weight matrix for the LSTM layer. The input weight
        %   matrix is a vertical concatenation of the four "gate" input
        %   weight matrices in the forward pass of an LSTM. Those
        %   individual matrices are concatenated in the following order:
        %   input gate, forget gate, layer input, output gate. This matrix
        %   will have size 4*NumHiddenUnits-by-InputSize.
        InputWeights
        
        % RecurrentWeights   The recurrent weights for the layer
        %   The recurrent weight matrix for the LSTM layer. The recurrent
        %   weight matrix is a vertical concatenation of the four "gate"
        %   recurrent weight matrices in the forward pass of an LSTM. Those
        %   individual matrices are concatenated in the following order:
        %   input gate, forget gate, layer input, output gate. This matrix
        %   will have size 4*NumHiddenUnits-by-NumHiddenUnits.
        RecurrentWeights        
    end

    properties(SetAccess = private, Hidden, Dependent)
        % OutputSize   The number of hidden units in the layer. See
        % NumHiddenUnits.
        OutputSize

        % OutputState   The hidden state of the layer. See HiddenState.
        OutputState
    end

    methods
        function this = LSTMLayerClass(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function val = get.InputWeights(this)
            val = this.PrivateLayer.InputWeights.HostValue;
            if isdlarray(val)
                val = extractdata(val);
            end   
        end
        
        function this = set.InputWeights(this, value)
            if isequal(this.InputSize, 'auto')
                expectedInputSize = NaN;
            else
                expectedInputSize = this.InputSize;
            end
            attributes = {'size', [4*this.NumHiddenUnits expectedInputSize],...
                'real', 'nonsparse'};
            value = iGatherAndValidateParameter(value, attributes);
            
            if ~isempty(value)
                inputs = {iMakeSizeOnlyArray([size(value,2) NaN NaN],'CBT')};
                if this.HasStateInputs
                    stateBlank = iMakeSizeOnlyArray([this.NumHiddenUnits NaN],'CB');
                    inputs(2:3) = {stateBlank stateBlank};
                end
                this.PrivateLayer = this.PrivateLayer.configureForInputs(inputs);
            end
            this.PrivateLayer.InputWeights.Value = value;
        end
        
        function val = get.RecurrentWeights(this)
            val = this.PrivateLayer.RecurrentWeights.HostValue;
            if isdlarray(val)
                val = extractdata(val);
            end            
        end
        
        function this = set.RecurrentWeights(this, value)
            attributes = {'size', [4*this.NumHiddenUnits this.NumHiddenUnits],...
                'real', 'nonsparse'};
            value = iGatherAndValidateParameter(value, attributes);
            this.PrivateLayer.RecurrentWeights.Value = value;
        end
        
        function val = get.OutputSize(this)
            val = this.NumHiddenUnits;
        end

        function val = get.OutputState(this)
            val = this.HiddenState;
        end

        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 5.0;
            out.Name = privateLayer.Name;
            out.InputSize = privateLayer.InputSize;
            out.NumHiddenUnits = privateLayer.HiddenSize;
            out.ReturnSequence = privateLayer.ReturnSequence;
            out.StateActivationFunction = privateLayer.Activation;
            out.GateActivationFunction = privateLayer.RecurrentActivation;
            out.HasStateInputs = privateLayer.HasStateInputs;
            out.HasStateOutputs = privateLayer.HasStateOutputs;
            out.InputWeights = toStruct(privateLayer.InputWeights);
            out.RecurrentWeights = toStruct(privateLayer.RecurrentWeights);
            out.Bias = toStruct(privateLayer.Bias);
            if privateLayer.HasStateInputs
                out.CellState = [];
                out.HiddenState = [];
            else
                out.CellState = toStruct(privateLayer.CellState);
                out.HiddenState = toStruct(privateLayer.HiddenState);
            end
            out.InitialCellState = gather(privateLayer.InitialCellState);
            out.InitialHiddenState = gather(privateLayer.InitialHiddenState);
        end
    end
   
    % Reorder properties
    methods (Hidden)
        function displayAllProperties(this)
            proplist = properties( this );
            % Function to reorder property display
            proplist = this.reorderProperties(proplist);
            matlab.mixin.CustomDisplay.displayPropertyGroups( ...
                this, ...
                this.propertyGroupGeneral( proplist ) );
        end

        function propList = reorderProperties(~, propList)
            nameIdx = 3;
            scalarSizeIdx = 4:5;
            outputModeIdx = 6;
            activationFunctionIdx = 7:8;
            hasStateIdx = 9:10;
            inputWeightIdx = [1,11:13];
            recurrentWeightIdx = [2,14:16];
            biasIdx = 17:20;
            statesNamesInputsOutputs = 21:26;
            idx = [nameIdx, scalarSizeIdx, outputModeIdx, activationFunctionIdx, ...
                hasStateIdx, inputWeightIdx, recurrentWeightIdx, biasIdx, ...
                statesNamesInputsOutputs]; 
            propList = propList(idx);
        end
    end

    methods(Static)
        function this = loadobj(in)
            if in.Version <= 1.0
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            if in.Version <= 2.0
                in = iUpgradeVersionTwoToVersionThree(in);
            end
            if in.Version <= 3.0
                in = iUpgradeVersionThreeToVersionFour(in);
            end
            if in.Version <= 4.0
                in = iUpgradeVersionFourToVersionFive(in);
            end
            internalLayer = nnet.internal.cnn.layer.LSTM( in.Name, ...
                in.InputSize, ...
                in.NumHiddenUnits, ...
                true, ...
                true, ...
                in.ReturnSequence, ...
                in.StateActivationFunction, ...
                in.GateActivationFunction, ...
                in.HasStateInputs, ...
                in.HasStateOutputs );
            internalLayer.InputWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.InputWeights);
            internalLayer.RecurrentWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.RecurrentWeights);
            internalLayer.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Bias);
            if ~in.HasStateInputs
                internalLayer.CellState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter.fromStruct(in.CellState);
                internalLayer.HiddenState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter.fromStruct(in.HiddenState);
            end
            internalLayer.InitialHiddenState = in.InitialHiddenState;
            internalLayer.InitialCellState = in.InitialCellState;

            this = nnet.cnn.layer.LSTMLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(obj)
            description = iGetMessageString( ...
                'nnet_cnn:layer:LSTMLayer:oneLineDisplay', ...
                num2str(obj.NumHiddenUnits));
            
            type = iGetMessageString( 'nnet_cnn:layer:LSTMLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            generalParameters = { 'Name', ...
                'InputNames', ...
                'OutputNames', ...
                'NumInputs', ...
                'NumOutputs', ...
                'HasStateInputs', ...
                'HasStateOutputs' };
            hyperParameters = { 'InputSize', ...
                'NumHiddenUnits', ...
                'OutputMode', ...
                'StateActivationFunction', ...
                'GateActivationFunction' };
            learnableParameters = { 'InputWeights', ...
                'RecurrentWeights', ...
                'Bias' };
            stateParameters = { 'HiddenState', 'CellState' };
            groups = [
                this.propertyGroupGeneral( generalParameters )
                this.propertyGroupHyperparameters( hyperParameters )
                this.propertyGroupLearnableParameters( learnableParameters )
                this.propertyGroupDynamicParameters( stateParameters )
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
% iUpgradeVersionOneToVersionTwo   Upgrade a v1 (R2017b) saved struct to a
% v2 saved struct. This means transforming the cell state and hidden state
% dynamic parameters into struct() format. Additionally, OutputState and
% OutputSize are re-named to HiddenState and NumHiddenUnits.
S.Version = 2;
S.CellState = toStruct(S.CellState);
S.HiddenState = toStruct(S.OutputState);
S.NumHiddenUnits = S.OutputSize;
end

function S = iUpgradeVersionTwoToVersionThree(S)
% iUpgradeVersionTwoToVersionThree   Upgrade a v2 (R2018a) saved struct to
% a v3 saved struct. This means adding the state activation and gate
% activation properties to the layer.
S.Version = 3;
S.StateActivationFunction = 'tanh';
S.GateActivationFunction = 'sigmoid';
end

function S = iUpgradeVersionThreeToVersionFour(S)
% Upgrade a v3 (R2018b) saved struct to a v4 saved struct. This means
% adding input weights, recurrent weights and bias initializers set to 
% 'narrow-normal', 'narrow-normal', and 'unit-forget-gate'.
S.Version = 4;
S.InputWeights = iAddInitializerToLearnable(S.InputWeights, "Normal", []);
S.RecurrentWeights = iAddInitializerToLearnable(S.RecurrentWeights, "Normal", []);
S.Bias = iAddInitializerToLearnable(S.Bias, "UnitForgetGate", {{'LSTM'}});
end

function S = iUpgradeVersionFourToVersionFive(S)
% Upgrade a v4 (R2020a) saved struct to a v5 saved struct. This means
% adding HasStateInputs and HasStateOutputs Booleans.
S.Version = 5;
S.HasStateInputs = false;
S.HasStateOutputs = false;
end

function s = iAddInitializerToLearnable(s, name, arguments)
s.Initializer = struct('Class', ...
    "nnet.internal.cnn.layer.learnable.initializer."+name, ...
    'ConstructorArguments', arguments);
end

function value = iGatherAndValidateParameter(varargin)
try
    value = nnet.internal.cnn.layer.paramvalidation...
        .gatherAndValidateNumericParameter(varargin{:});
catch exception
    throwAsCaller(exception)
end
end

function dlX = iMakeSizeOnlyArray(varargin)
dlX = deep.internal.PlaceholderArray(varargin{:});
end