% PROJECTE VISIÓ PER COMPUTADOR - DETECCIÓ DE SENYALS
% Fitxer principal (Main)

clear; clc; close all;
addpath('src'); % Afegir la carpeta de funcions al path

%% 1. CONFIGURACIÓ I DADES
datasetPath = 'imatges_senyals';
config.fourierDesc = 10;
config.colorBins = 16;
config.shapeFeats = 6;
config.imgSize = 64;
config.cellSize = 8;

fprintf('Carregant dataset...\n');
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

%% 2. EXTRACCIÓ DE CARACTERÍSTIQUES (TRAIN)
fprintf('Processant TRAIN set...\n');
[XTrain, YTrain, trainMu, trainSigma] = extractDatasetFeatures(imdsTrain, config, [], []); 
% Nota: Passem [] a mu/sigma perquè els calculi de zero

%% 3. ENTRENAMENT
fprintf('Entrenant models...\n');
models = trainModels(XTrain, YTrain);

%% 4. AVALUACIÓ (TEST)
fprintf('Processant TEST set...\n');
% Important: Usem la mu i sigma del TRAIN per normalitzar el TEST
[XTest, YTest] = extractDatasetFeatures(imdsTest, config, trainMu, trainSigma);

% Prediccions i precisió
[results, bestModelName, bestIdx] = evaluateModels(models, XTest, YTest);

%% 5. VISUALITZACIÓ
visualizeResults(results, bestIdx, models, imdsTest, YTest);

%% 6. GUARDAR MODEL PER APLICACIÓ
fprintf('Guardant el millor model per a ús futur...\n');

% Seleccionem només l'objecte del model guanyador
switch bestIdx
    case 1, finalModel = models.svm;
    case 2, finalModel = models.rf;
    case 3, finalModel = models.knn;
end

% Guardem el model, la configuració i les dades de normalització (mu/sigma)
save('model_entrenat.mat', 'finalModel', 'bestModelName', 'config', 'trainMu', 'trainSigma');
fprintf('Model guardat correctament a "model_entrenat.mat".\n');