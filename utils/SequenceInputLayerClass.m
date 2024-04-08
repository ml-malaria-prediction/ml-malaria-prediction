classdef SequenceInputLayerClass < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable & ...
        nnet.cnn.layer.mixin.NormalizableInput & ...
        nnet.internal.cnn.layer.Summarizable
    % SequenceInputLayer   Sequence input layer
    %
    %   To create a sequence input layer, use sequenceInputLayer
    %
    %   SequenceInputLayer properties:
    %       Name                        - A name for the layer.
    %       InputSize                   - The size of the input.
    %       Normalization               - Data normalization applied when
    %                                     data is forward propagated
    %                                     through the input layer. Valid
    %                                     options are 'zerocenter',
    %                                     'zscore', 'rescale-symmetric',
    %                                     'rescale-zero-one', 'none', or a
    %                                     function handle.
    %       NormalizationDimension      - Dimension over which the same
    %                                     normalization is applied. Valid
    %                                     values are 'auto', 'channel',
    %                                     'element', 'all'.
    %       Mean                        - The mean value used for zero
    %                                     centering and z-score normalization.
    %                                     The same value is used for all
    %                                     time steps.
    %       StandardDeviation           - The standard deviation used for
    %                                     z-score normalization. The same
    %                                     value is used for all time steps.
    %       Min                         - The minimum value used for
    %                                     rescaling. The same value is used
    %                                     for all time steps.
    %       Max                         - The maximum value used for
    %                                     rescaling. The same value is used
    %                                     for all time steps.
    %       MinLength                   - Minimum sequence length the input 
    %                                     layer accepts, specified as a 
    %                                     positive integer. 
    %       SplitComplexInputs          - Flag indicating whether the layer 
    %                                     should split the real and
    %                                     imaginary components of the data. 
    %       NumOutputs                  - The number of outputs of the
    %                                     layer.
    %       OutputNames                 - The names of the outputs of the
    %                                     layer.
    %
    %   Example:
    %       Create a sequence input layer to accept a multi-dimensional
    %       time series with 5 values per timestep
    %
    %       layer = sequenceInputLayer(5)
    %
    %   See also sequenceInputLayer
    
    %   Copyright 2017-2022 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        % InputSize   Size of the input data as an integer.
        InputSize
        
        % Normalization   A string that specifies the normalization applied
        %                 to the input data every time it is forward
        %                 propagated through the input layer. Valid values
        %                 are 'zerocenter', 'zscore', 'rescale-symmetric'
        %                 'rescale-zero-one', or 'none'. This property is
        %                 read-only.
        Normalization
        
        % MinLength       The minimum sequence length the input layer
        % accepts.
        MinLength

        % SplitComplexInputs       If true, layer splits the real and 
        % imaginary components of the input data and outputs them as
        % separate channels.
        SplitComplexInputs
    end
    
    properties(Dependent)
        % Mean    The mean of the training data, used for 'zerocenter'
        %         and 'zscore' normalization.
        Mean
        
        % StandardDeviation    The standard deviation of the training data,
        %                      used for 'zscore' normalization.
        StandardDeviation
        
        % Min    The minimum of the training data, used for
        %        'rescale-symmetric' and 'rescale-zero-one' normalization.
        Min
        
        % Max    The maximum of the training data, used for
        %        'rescale-symmetric' and 'rescale-zero-one' normalization.
        Max
        
        % NormalizationDimension   Dimension over which the same 
        %                          normalization is applied. Valid values
        %                          are 'auto', 'channel', 'element', 'all'.
        NormalizationDimension
    end
    
    methods
        function this = SequenceInputLayerClass(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            out.Version = 5.0;
            out.Name = this.PrivateLayer.Name;
            out.InputSize = this.PrivateLayer.InputSize;
            out.Normalization = iSaveTransforms(this.PrivateLayer.Normalization);
            out.NormalizationDimension = this.PrivateLayer.NormalizationDimension;
            out.MinLength = this.PrivateLayer.MinLength;
            out.SplitComplexInputs = this.PrivateLayer.SplitComplexInputs;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.InputSize(this)
            val = this.PrivateLayer.InputSize;
        end
        
        function val = get.Normalization(this)
            transformObj = this.PrivateLayer.Normalization;
            if isempty(transformObj)
                val = 'none';
            elseif transformObj.Type == "rescale"
                if isequal( [transformObj.TargetMin, transformObj.TargetMax], [0 1] )
                    val = 'rescale-zero-one';
                elseif isequal( [transformObj.TargetMin, transformObj.TargetMax], [-1 1] )
                    val = 'rescale-symmetric';
                end
            elseif transformObj.Type == "custom"
                val = transformObj.Function;
            else
                val = transformObj.Type;
            end
        end
        
        function val = get.NormalizationDimension(this)
            val = this.PrivateLayer.NormalizationDimension;
        end
        
        function this = set.NormalizationDimension(this, val)
            iAssertValidNormalizationDimension(val,this.Normalization);
            % The shape of statistics must be consistent with the
            % normalization dimension parameter
            statistics = getNormalizationHyperParams(this);
            for i = 1:numel(statistics)
                allowComplexStatistic = (statistics{i} == "Mean");
                iAssertValidStatistics(this.(statistics{i}), statistics{i}, ...
                    this.Normalization, val, this.InputSize, allowComplexStatistic);
            end
            this.PrivateLayer.NormalizationDimension = convertStringsToChars(val);
        end
        
        function val = get.MinLength(this)
            val = this.PrivateLayer.MinLength;
        end

        function val = get.SplitComplexInputs(this)
            val = this.PrivateLayer.SplitComplexInputs;
        end
        
        function val = get.Mean(this)
            val = this.PrivateLayer.Mean;
        end
        
        function this = set.Mean(this, val)
            allowComplexStatistic = true;
            iAssertValidStatistics(val, 'Mean', this.Normalization, ...
                this.NormalizationDimension, this.InputSize, ...
                allowComplexStatistic, this.SplitComplexInputs);

            % Store as single
            this.PrivateLayer.Mean = single(gather(val));
        end
        
        function val = get.StandardDeviation(this)
            val = this.PrivateLayer.Std;
        end
        
        function this = set.StandardDeviation(this, val)
            iAssertValidStatistics(val, 'StandardDeviation', this.Normalization, ...
                this.NormalizationDimension, this.InputSize);
            this.PrivateLayer.Std = single(gather(val));
        end
        
        function val = get.Min(this)
            val = this.PrivateLayer.Min;
        end
        
        function this = set.Min(this, val)
            iAssertValidStatistics(val, 'Min', this.Normalization, ...
                this.NormalizationDimension, this.InputSize);
            this.PrivateLayer.Min = single(gather(val));
        end
        
        function val = get.Max(this)
            val = this.PrivateLayer.Max;
        end
        
        function this = set.Max(this, val)
            iAssertValidStatistics(val, 'Max', this.Normalization, ...
                this.NormalizationDimension, this.InputSize);
            this.PrivateLayer.Max = single(gather(val));
        end
    end
    
    methods(Static)
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
            if in.Version <= 4
                in = iUpgradeVersionFourToVersionFive(in);
            end
            internalLayer = nnet.internal.cnn.layer.SequenceInput( ...
                in.Name, in.InputSize, iLoadTransforms( in.Normalization ), ...
                in.MinLength, in.SplitComplexInputs );
            internalLayer.NormalizationDimension = in.NormalizationDimension;
            this = nnet.cnn.layer.SequenceInputLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            inputSizeString = iGetSizeString( this.InputSize );
            description = iGetMessageString( ...
                'nnet_cnn:layer:SequenceInputLayer:oneLineDisplay', ....
                inputSizeString );
            
            type = iGetMessageString( 'nnet_cnn:layer:SequenceInputLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            generalParameters = {
                'Name'
                'InputSize'
                'MinLength'
                'SplitComplexInputs'
                };
            
            hyperParameters = [
                {'Normalization'
                'NormalizationDimension'}
                getNormalizationHyperParams(this)
                ];
            
            groups = [
                this.propertyGroupGeneral( generalParameters )
                this.propertyGroupHyperparameters( hyperParameters )
                ];
        end

        function summary = getOneLineSummary(this)
            sizeString = iGetSizeString(this.InputSize);
            summary = getString(message('nnet_cnn:layer:SequenceInputLayer:oneLineDisplay',sizeString));
        end
    end

    methods(Access=private)
        function proplist = getNormalizationHyperParams(this)
            layerProperties = nnet.internal.cnn.layer.util.getPublicVisibleProperties(this);
            if isempty(this.PrivateLayer.Normalization)
                internalNormParams = {};
            else
                internalNormParams = this.PrivateLayer.Normalization.Hyperparams;
            end
            proplist = intersect(layerProperties,internalNormParams);
            proplist = proplist(:);
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function iAssertValidLayerName(name)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLayerName(name));
end

function S = iSaveTransforms(transforms)
% iSaveTransforms   Save a vector of transformations in the form of an
% array of structures
S = arrayfun( @serialize, transforms );
end

function transforms = iLoadTransforms( S )
% iLoadTransforms   Load a vector of transformations from an array of
% structures S
transforms = nnet.internal.cnn.layer.InputTransform.empty();
for i=1:numel(S)
    transforms = horzcat(transforms, iLoadTransform( S(i) )); %#ok<AGROW>
end
end

function transform = iLoadTransform(S)
% iLoadTransform   Load a transformation from a structure S
transform = nnet.internal.cnn.layer.InputTransformFactory.deserialize( S );
end

function iAssertValidNormalizationDimension(val,normalization)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateNormalizationDimension(val,normalization) );
end

function iAssertValidStatistics(value, valueName, normalization, normalizationDim, ...
    inputSize, allowComplexStatistic, splitComplexInputs)

if nargin < 6
    allowComplexStatistic = false;
end

if nargin < 7
    splitComplexInputs = false;
end

hasRowVectorStats = false;
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateNormalizationStatistics(...
    value, valueName, normalization, normalizationDim, inputSize, ...
    'HasRowVectorStats', hasRowVectorStats, ...
    'AllowComplexStatistic', allowComplexStatistic, ...
    'SplitComplexInputs', splitComplexInputs));
end
function iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    func();
catch exception
    throwAsCaller(exception)
end
end

function sizeString = iGetSizeString( sizeVector )
sizeString = nnet.internal.cnn.util.sizeToString( sizeVector );
end

function S = iUpgradeVersionOneToVersionTwo(S)
S.Version = 2;
S.Normalization = nnet.internal.cnn.layer.InputTransform.empty();
end

function S = iUpgradeVersionTwoToVersionThree(S)
% iUpgradeVersionTwoToVersionThree   Upgrade a v2 (2019a) saved struct to
% a v3 saved struct. This means adding the NormalizationDimension parameter.
S.Version = 3.0;
S.NormalizationDimension = 'auto';
end

function S = iUpgradeVersionThreeToVersionFour(S)
% iUpgradeVersionThreeToVersionFour   Upgrade a v3 (2021a) saved struct to
% a v4 saved struct. This means adding the MinLength parameter.
S.Version = 4.0;
S.MinLength = 1;
end

function S = iUpgradeVersionFourToVersionFive(S)
% iUpgradeVersionFourToVersionFive   Upgrade a v4 (2022a) saved struct to
% a v5 saved struct. This means adding the SplitComplexInputs parameter.
S.Version = 5.0;
S.SplitComplexInputs = false;
end
