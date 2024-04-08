rTimes=0;
X=[X; repmat(X((y>0),:,:),[rTimes,1,1]).*(1+randn([rTimes,1,1].*size(X((y>0),:,:)))/10)];
y2=[y; repmat(y(y>0),[rTimes,1])];
Y=y2;