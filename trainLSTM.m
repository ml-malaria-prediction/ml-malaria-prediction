folder = fileparts(which('trainLSTM.m'));
addpath(genpath(folder));

clear all


load('data/ParamsTrain.mat')
load('data/DataTrain.mat')

computeStatsTrain
computeRelevance=0;


prm.vx = [0.325 0.325 2];
prm.normVx=prm.vx./min(prm.vx);

X=squeeze(blobDataNan(:,:,:));
Y=y;

numPos=length(Y(Y>0));
numNeg=length(Y(Y<0));
cvalNum=5;
rng(93);
c = cvpartition(Y,'KFold',cvalNum);

X(isnan(X))=0;

X_tst=X(c.test(1),:,:);

classes = [categorical(-1) categorical(1)];      
X=squeeze(X(:,:,paramsIdx));

%%
nL=size(X,2);

for j=1:c.NumTestSets
    X_tr=X(c.training(j),:,:);
    Y_tr=categorical(Y(c.training(j)));
    clear X_trg
    for i=1:size(X_tr,1)
        X_trg{i}=squeeze(X_tr(i,1:nL,:))';
    end
    X_tr=X_trg';

    X_tst=X(c.test(j),:,:);
    Y_tst=categorical(Y(c.test(j)));
    clear X_test
    for i=1:size(X_tst,1)
        X_test{i}=squeeze(X_tst(i,1:nL,:))';
    end
    X_test=X_test';
   
    options = trainingOptions(optimizer, ...
    'ExecutionEnvironment','gpu', ...
        'GradientThreshold',GradTh, ... 
        'InitialLearnRate',ILR, ...
        'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
     'ValidationData',{X_test,Y_tst},...
    'Plots',"none");

    inputSize = size(X_tr{1},1);
 
    layers = [ ...
        sequenceInputLayer(inputSize,'Normalization','zerocenter');
        lstmLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(drp)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer('Classes',classes)];

    [net,info] = trainNetwork(X_trg,Y_tr,layers,options);
    netS{j}=net;
     YPred = classify(net,X_test, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest');

    accR(j) = sum(YPred == Y_tst)./numel(Y_tst);
end

acc=zeros(cvalNum,size(X_tst,2)-1);
accN=zeros(cvalNum,size(X_tst,2)-1);
accP=zeros(cvalNum,size(X_tst,2)-1);
accMean=zeros(cvalNum,size(X_tst,2)-1);

computeStatsTrain

X=squeeze(blobDataNan(:,:,:));
X=squeeze(X(:,:,paramsIdx));
X(isnan(X))=0;

for j=1:cvalNum
   
    Y_tr=categorical(Y(c.training(j)));

    rRange=c.test(j);
    rRange=rRange(1:length(y));
    X_tst=X(rRange,:,:);
    Y_tst=categorical(Y(rRange));
 

    Ytst=zeros(size(X_tst,2)-1,sum(rRange));
    Scores=zeros(size(X_tst,2)-1,sum(rRange));
    for iL=1:size(X_tst,2)-1
        clear X_test
        clear X_test2
        for i=1:size(X_tst,1)
            X_test{i}=squeeze(X_tst(i,1:end-iL+1,:))';
        end

        X_test=X_test';
        net=netS{j};
        YPred = classify(net,X_test, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest');
         scores = predict(net,X_test, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest','ReturnCategorical',0);
        acc(j,iL) = sum(YPred == Y_tst)./numel(Y_tst);
        Y_tst2=double(string(Y_tst));
        YPred2=double(string(YPred));
    

        y2=double(string(Y_tst));
        y2(y2==-1)=0;
        y2(:,2)=~y2;
        y2(:,[2,1])=y2;

        accN(j,iL)=sum(YPred2(Y_tst2<0)==Y_tst2(Y_tst2<0))/numel(Y_tst2(Y_tst2<0));
        accP(j,iL)=sum(YPred2(Y_tst2>0)==Y_tst2(Y_tst2>0))/numel(Y_tst2(Y_tst2>0));
        accMean(j,iL)=(accN(j,iL)+accP(j,iL))/2;

       

        Ytst(iL,:)=Y_tst2;
        Scores(iL,:)=scores(:,2);
        C = confusionmat(Y_tst2,YPred2);
        cAll(j,iL,:,:)=C;
        [pr,rec,f1,meanF1]=PRcurves(C);
        [xRocAll{j,iL},yRocAll{j,iL},th,AUCAll(j,iL)] = perfcurve(Y_tst2,scores(:,2),1);

        %compute parameter relevance
        if computeRelevance==1
            X_testA=zeros([size(X_test,1),size(X_test{1})]);
            for ii=1:size(X_test,1)
                    X_testA(ii,:,:)=squeeze((X_test{ii}));
            end
            for iF=1:size(X_tst,3)
                
                for iPerm=1:10
                    rng(iPerm);
                    idx = randperm(size(X_test,1));
                    X_testP=X_testA;
                    X_testP(:,iF,:)=X_testA(idx,iF,:);
                    for ii=1:size(X_test,1)
                        X_test2{ii}=squeeze(X_testP(ii,:,:));
                    end
                    YPred = classify(net,X_test2, ...
                     'MiniBatchSize',miniBatchSize, ...
                    'SequenceLength','longest');
                    accLoss(j,iL,iF,iPerm)=acc(j,iL)-sum(YPred == Y_tst)./numel(Y_tst);
    
                       
                end
    
            end
        end
    end
    
end



