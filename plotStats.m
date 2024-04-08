load('DataTrain.mat')
computeStatsTrain

meanParamsP=squeeze(nanmean(blobDataNan(y>0,:,:),1));
stdParamsP=squeeze(nanstd(blobDataNan(y>0,:,:),1));

meanParamsN=squeeze(nanmean(blobDataNan(y<=0,:,:),1));
stdParamsN=squeeze(nanstd(blobDataNan(y<=0,:,:),1));

cLP=meanParamsP-2*stdParamsP;
cHP=meanParamsP+2*stdParamsP;
cLN=meanParamsN-2*stdParamsN;
cHN=meanParamsN+2*stdParamsN;

cLP=squeeze(nanmin(blobDataNan(y>0,:,:),[],1));
cHP=squeeze(nanmax(blobDataNan(y>0,:,:),[],1));
cLN=squeeze(nanmin(blobDataNan(y<=0,:,:),[],1));
cHN=squeeze(nanmax(blobDataNan(y<=0,:,:),[],1));

perc=95;

cLP = squeeze(prctile(blobDataNan(y>0,:,:),100-perc));
cHP = squeeze(prctile(blobDataNan(y>0,:,:),perc));
cLN = squeeze(prctile(blobDataNan(y<=0,:,:),100-perc));
cHN = squeeze(prctile(blobDataNan(y<=0,:,:),perc));

xData=(33-1+(1:size(blobDataNan,2)))';
plotParams=[1:19];

for i=1:length(plotParams)
    j=plotParams(i);
     g=figure();
    g.Position([3,4])=[406, 340];
     plot(xData,meanParamsP(:,j),'b');
    hold on; 
    plot(xData,meanParamsN(:,j),'r');
    hold on;
    
    ciplot(cLP(:,j),cHP(:,j),xData,'b');
    hold on; 
    ciplot(cLN(:,j),cHN(:,j),xData,'r');
    xlabel('hours post infection')
    ylabel(blobParamsNamesUm{plotParams(i)});
    if i==1
        legend('Positive class','Negative class','Positive range','Negative range');
    end
end
