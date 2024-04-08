load('data/DataTest.mat')
load('data/nnets.mat')

computeStatsTest
y(y==-1)=0;

X_tst=squeeze(blobDataNan(:,:,:));
paramsRelevance=0;

X_tst(isnan(X_tst))=0;
X_tst=X_tst(:,:,paramsIdx);
Y_tst=categorical(y);


%%
clear acc accMeanT 

clear xRocV
clear yRocV
clear aucV
clear xRoc yRoc 
clear YPredV
for j=1:c.NumTestSets
    for iL=1:size(X_tst,2)-1
        clear X_test
        for i=1:size(X_tst,1)
            X_test{i}=squeeze(X_tst(i,1:end-iL+1,:))';
        end

        X_test=X_test';
        net=netS{j};
        YPred = predict(net,X_test, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest','ReturnCategorical',0);
         
        thF=0.5;

        Y_tst2=double(string(Y_tst));

        YPred=YPred(:,2);
        if iL
            [xRoc,yRoc,th,AUC,OPTROCPT] = perfcurve(Y_tst2,YPred,1);
            xRocV{j,iL}=xRoc;
            yRocV{j,iL}=yRoc;
            aucV(j,iL)=AUC;
        end

        YPredV(iL,:)=YPred;

        YPred(YPred<thF)=0;
        YPred(YPred>=thF)=1;
       
        Y_tst2=double(string(Y_tst));
        accN(j,iL)=sum(YPred(Y_tst2==0)==Y_tst2(Y_tst2==0))/numel(Y_tst2(Y_tst2==0));
        accP(j,iL)=sum(YPred(Y_tst2>0)==Y_tst2(Y_tst2>0))/numel(Y_tst2(Y_tst2>0));
        accMeanT(j,iL)=(accN(j,iL)+accP(j,iL))/2;
       
   end
     
end
