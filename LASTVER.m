

%  maxminT - input file.
%  it is contain 3 columns , first column contains MaxTempretuer 
%  seconed column contains mintempretuer and the third column ontain Humidity 

%   the output from network is filr energy_median that contains energy consumption 
%  for one house from 12/10/2012 until 28/2/2014
 

X = tonndata(maxminT,false,false);
%T = tonndata(energy_median,false,false);

% we will read first block
allb0=xlsread('C:\Users\mahmad\Desktop\WP\SML\daily_dataset\daily_dataset\block_0.csv');

%import date for all block 
day = importfile('C:\Users\mahmad\Desktop\WP\SML\daily_dataset\daily_dataset\block_0.csv',2, 25575);

[mydata,textdata]=xlsread('C:\Users\mahmad\Desktop\WP\SML\daily_dataset\daily_dataset\block_0.csv');

string day[];


% this code will ask user to enter the ID of house in block 0

str1 = input('enter ID','s')
%str = 'MAC003863';
j=1;
flage =0;
m=0;

while (flage == 0)
     if (strcmp(str1(:,:),textdata(j,:))==1) 
         f=1;
     end
     %if (strcmp(day(j),day(1))) 
     if (day(j)==day(1))
         m=1;
     end
   
     if ((m==1) && (f==1))

     energy_median(:,:)=allb0(1+j:505+j,1); 
     flage =1;

     else j=j+1;
      end
end 
          




% 'trainlm' is training function 
trainFcn = 'trainlm';  

% Create a Network 
% we detect idden layers equals tp 100 layers
inputDelays = 1:2;
feedbackDelays = 1:2;
hiddenLayerSize = 100;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);


net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for Training and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer
% states. Using PREPARETS allows you to keep your original time series data
% unchanged, while easily customizing it for networks with differing
% numbers of delays, with open loop or closed loop feedback modes.
[x,xi,ai,t] = preparets(net,X,{},T);

% Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
net.performFcn = 'mse';  

% % Choose Plot Functions
% % For a list of all plot functions type: help nnplot
% net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
%     'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};

% Train the Network
[net,tr] = train(net,x,t,xi,ai);

% Test the Network
y = net(x,xi,ai);
e = gsubtract(t,y);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(t,tr.trainMask);
valTargets = gmultiply(t,tr.valMask);
testTargets = gmultiply(t,tr.testMask);
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)



% Closed Loop Network

netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
view(netc)
[xc,xic,aic,tc] = preparets(netc,X,{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc)

% Multi-step Prediction

numTimesteps = size(x,2);
knownOutputTimesteps = 1:(numTimesteps-7);
predictOutputTimesteps = (numTimesteps-6):numTimesteps;
X1 = X(:,knownOutputTimesteps);
T1 = T(:,knownOutputTimesteps);
[x1,xio,aio] = preparets(net,X1,{},T1);
[y1,xfo,afo] = net(x1,xio,aio);



x2 = X(1,predictOutputTimesteps);
[netc,xic,aic] = closeloop(net,xfo,afo);
[y2,xfc,afc] = netc(x2,xic,aic);
multiStepPerformance = perform(net,T(1,predictOutputTimesteps),y2)

disp(y2)
