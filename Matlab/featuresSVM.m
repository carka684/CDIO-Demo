% Testar HOG-feature extraction på Horses dataset
clear
close all;
addpath('libsvm-3.18/matlab/');
addpath('efficientLBP/');

% Read in images of horses and not horses
numPos = 200;
numNeg = 400;
numSamples = numPos + numNeg;

horses = zeros(512, 512, numPos);
notHorses = zeros(512, 512, numNeg);

for i = 1:numPos
    imHorse = rgb2gray(imread(strcat('Images/Horse/horse_', num2str(i) , '.jpg')));
    imHorse = imresize(imHorse, [512, 512]);
    horses(:,:,i) = imHorse;
end

for i = 1:numNeg
    imNotHorse = rgb2gray(imread(strcat('Images/NotHorse/nothorse_', num2str(i) , '.jpg')));
    imNotHorse = imresize(imNotHorse, [512, 512]);
    notHorses(:,:,i) = imNotHorse;
end

% figure()
% colormap(gray);
% imagesc(horses(:,:,3));
% 
% figure()
% colormap(gray);
% imagesc(notHorses(:,:,3));

% Create training data and test data with corresponding label vectors
% Change all num depending on numSamples
numTrainPos = 140;
numTrainNeg = 280;
numTrainSamples = numTrainPos + numTrainNeg;
trainImages = cat(3, horses(:,:, 1:numTrainPos), notHorses(:,:, 1:numTrainNeg));
trainLabels = [ones(1, numTrainPos), -ones(1, numTrainNeg)]';

numTestPos = 40;
numTestNeg = 80;
numTestSamples = numTestPos + numTestNeg;
testImages = cat(3, horses(:,:, numTrainPos+1:numTrainPos+numTestPos), notHorses(:,:, numTrainNeg+1:numTrainNeg+numTestNeg));
testLabels = [ones(1, numTestPos), -ones(1, numTestNeg)]';

numValidationPos = 20;
numValidationNeg = 40;
numValidationSamples = numValidationPos + numValidationNeg;
validationImages = cat(3, horses(:,:, numPos-numValidationPos+1:end), notHorses(:,:, numNeg-numValidationNeg+1:end));
validationLabels = [ones(1, numValidationPos), -ones(1, numValidationNeg)]';

%% Extract features
% Choose featureType

featureType = 'LBP';

switch(featureType)
    case 'HOG'
        featureLength = length(extractHOGFeatures(trainImages(:,:,1), 'cellsize' ,[16 16]));
        trainFeatures = zeros(numTrainSamples, featureLength);
        testFeatures = zeros(numTestSamples, featureLength);
        validationFeatures = zeros(numValidationSamples, featureLength);
        
        for i = 1:numTrainSamples
            [trainFeatures(i,:)] = extractHOGFeatures(trainImages(:,:,i), 'cellsize', [16 16]);
        end
        
        for i = 1:numTestSamples
            [testFeatures(i,:)] = extractHOGFeatures(testImages(:,:,i), 'cellsize', [16 16]);
        end
        
        for i = 1:numValidationSamples
            [validationFeatures(i,:)] = extractHOGFeatures(validationImages(:,:,i), 'cellsize', [16 16]);
        end
        
    case 'LBP'
        for i = 1:numTrainSamples
            filtR=generateRadialFilterLBP(8, 1);
            effLBP = efficientLBP(trainImages(:,:,i), 'filtR', filtR, 'isRotInv', true, 'isChanWiseRot', true);
            histVectorTrain(:,i) = hist(effLBP(:),0:255);
        end
        
        for i = 1:numTestSamples
            filtR=generateRadialFilterLBP(8, 1);
            effLBP = efficientLBP(testImages(:,:,i), 'filtR', filtR, 'isRotInv', true, 'isChanWiseRot', true);
            histVectorTest(:,i) = hist(effLBP(:),0:255);
        end
        
        for i = 1:numValidationSamples
            filtR=generateRadialFilterLBP(8, 1);
            effLBP = efficientLBP(validationImages(:,:,i), 'filtR', filtR, 'isRotInv', true, 'isChanWiseRot', true);
            histVectorValidation(:,i) = hist(effLBP(:),0:255);
        end
        histVectorWhole = [histVectorTrain, histVectorTest, histVectorValidation];
        histVector = histVectorWhole';
        histVector(:,end) = [];
        histVectorTrain = histVector(1:numTrainSamples,:);
        histVectorTest = histVector(numTrainSamples+1:numTrainSamples+numTestSamples,:);
        histVectorValidation = histVector(end-numValidationSamples+1:end,:);
        histVector = (histVector - ones(size(histVector,1), 1) * mean(histVectorTrain, 1)) ./ (ones(size(histVector,1), 1) * var(histVectorTrain, 1));
        histVector(:,isnan(histVector(1,:))) =[];
        
        histVectorTrain = histVector(1:numTrainSamples,:);
        histVectorTest = histVector(numTrainSamples+1:numTrainSamples+numTestSamples,:);
        histVectorValidation = histVector(end-numValidationSamples+1:end,:);
        
        trainFeatures = histVectorTrain;
        testFeatures = histVectorTest;
        validationFeatures = histVectorValidation;
        
   figure()
   plot(histVector(:,16))

end

%% optimize parameter c
n = -20:20;
accuracy = nan(size(n));

for i = 1:length(n)
   c = 2^n(i);
     % create model
        model = svmtrain(trainLabels, trainFeatures, ['-c ' num2str(c)]); 
        
        [lbl, acc, dec] = svmpredict(testLabels, testFeatures, model, []);
        accuracy(i)= acc(1);
end

figure(1)
plot(n, accuracy)
xlabel('n'); ylabel('Accuracy'); title('Accuracy vs n');

%% Choose n (c) and validate

n = 14;
c = 2^n;

model = svmtrain(trainLabels, trainFeatures, ['-c ' num2str(c)]);

[lbl, acc, dec]= svmpredict(validationLabels, validationFeatures, model, []);

lbl
acc

% figure(2);
% colormap(gray);
% imagesc(validationImages(:,:,3));
% hold on;
% plot(hogVis(3));
% 
% figure(3);
% colormap(gray);
% imagesc(validationImages(:,:,4));
% hold on;
% plot(hogVis(4));

