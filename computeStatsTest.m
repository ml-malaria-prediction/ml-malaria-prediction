blobDataNames={'Volume','Mean Intensity','Max Intensity','Convex Envelope','X',...
    'Y','Elevation','Cell length','Cell width','Cell height',...
    'OX Orientation','OY Orientation','OZ Orientation',...
    'Cell fine elements','Length to width ratio','Distance traveled','Solidity','Ratio of volume to eq. diameter',...
    'Equivalent diameter'};


prm.vx = [0.325 0.325 2];
prm.normVx=prm.vx./min(prm.vx);

b=double(squeeze(blobData(:,:,[5,6,7])));
b(:,:,[1,2])=b(:,:,[1,2])*prm.vx(1);
b(:,:,3)=b(:,:,3).*prm.vx(3);
allBlobDist=zeros(size(blobData,[1,2]));
bb=sum(abs(b),3);
b(bb==0)=nan;
b=diff(b,1,2);
b=b.^2;
b=sum(b,3);
b=sqrt(b);
%b=mean(b,2);
allBlobDist(:,2:end)=cumsum(b,2);
blobData(:,:,16)=allBlobDist;


blobDataNan=blobData;
y=cellsGT;

blobDataNan(:,:,[1,4])=double(blobDataNan(:,:,[1,4]))*prm.vx(1)*prm.vx(2)*prm.vx(3);

blobDataNan(:,:,17)=double(blobDataNan(:,:,1))./double(blobDataNan(:,:,4));
blobDataNan(:,:,18)=double(blobDataNan(:,:,1))./nthroot(double(blobDataNan(:,:,4)),3);
blobDataNan(:,:,19)=nthroot(6/pi*double(blobDataNan(:,:,1)),3);


