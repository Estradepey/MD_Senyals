% PROJECTE VISIÓ PER COMPUTADOR - DETECCIÓ DE SENYALS DE TRÀNSIT
% Universitat Politècnica de Catalunya - FIB
% Millores sobre descriptors de Fourier bàsics

%% ============================================================================
%% 1. CÀRREGA I PREPARACIÓ DEL DATASET
%% ============================================================================
clear; clc; close all;

datasetPath = 'E:\MATLAB\MD_Senyals\imatges_senyals';
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split train/test
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

fprintf('Dataset carregat:\n');
fprintf('- Train: %d imatges\n', numel(imdsTrain.Files));
fprintf('- Test: %d imatges\n', numel(imdsTest.Files));
fprintf('- Classes: %d\n\n', numel(categories(imdsTrain.Labels)));

%% ============================================================================
%% 2. EXTRACCIÓ DE CARACTERÍSTIQUES - TRAIN
%% ============================================================================

% Configuració de característiques
config.fourierDesc = 20;        % Descriptors de Fourier
config.huMoments = 7;           % Moments de Hu
config.colorBins = 15;          % Bins per histograma de color (HSV)
config.shapeFeats = 6;          % Característiques de forma
config.imgSize = 64;            % Mida normalitzada

numFeatures = config.fourierDesc + config.huMoments + ...
              config.colorBins * 3 + config.shapeFeats;

fprintf('Extracting features from TRAIN set...\n');
[XTrain, trainValid] = extractFeatures(imdsTrain, config);
YTrain = imdsTrain.Labels(trainValid);
fprintf('Train features: %d imatges vàlides de %d\n\n', sum(trainValid), numel(trainValid));

%% ============================================================================
%% 3. NORMALITZACIÓ I PREPARACIÓ
%% ============================================================================

% Normalitzar característiques (important!)
% Usar normxcorr per evitar divisió per zero
mu = mean(XTrain, 1);
sigma = std(XTrain, 0, 1);

% Evitar divisió per zero: si sigma és 0, posar-lo a 1
sigma(sigma < 1e-10) = 1;

XTrain_norm = (XTrain - mu) ./ sigma;

% Eliminar NaN/Inf que puguin haver aparegut
validIdx = all(isfinite(XTrain_norm), 2);
XTrain_norm = XTrain_norm(validIdx, :);
YTrain = YTrain(validIdx);

fprintf('Després de neteja: %d mostres d''entrenament\n\n', size(XTrain_norm, 1));

%% ============================================================================
%% 4. ENTRENAMENT DE CLASSIFICADORS
%% ============================================================================

fprintf('=== ENTRENAMENT DE MODELS ===\n');

% 4.1 SVM amb validació creuada
fprintf('Entrenant SVM amb RBF kernel...\n');
t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);
mdl_svm = fitcecoc(XTrain_norm, YTrain, 'Learners', t, 'Coding', 'onevsall');

% Validació creuada per estimar precisió
cvModel_svm = crossval(mdl_svm, 'KFold', 5);
cvAcc_svm = 1 - kfoldLoss(cvModel_svm);
fprintf('  CV Accuracy: %.2f%%\n', cvAcc_svm * 100);

% 4.2 Random Forest
fprintf('Entrenant Random Forest...\n');
mdl_rf = TreeBagger(100, XTrain_norm, YTrain, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'MinLeafSize', 5);
oobError = oobError(mdl_rf);
fprintf('  OOB Error: %.2f%%\n', (1 - oobError(end)) * 100);

% 4.3 KNN
fprintf('Entrenant KNN...\n');
mdl_knn = fitcknn(XTrain_norm, YTrain, ...
    'NumNeighbors', 7, ...
    'Distance', 'euclidean', ...
    'Standardize', true);

cvModel_knn = crossval(mdl_knn, 'KFold', 5);
cvAcc_knn = 1 - kfoldLoss(cvModel_knn);
fprintf('  CV Accuracy: %.2f%%\n\n', cvAcc_knn * 100);

%% ============================================================================
%% 5. AVALUACIÓ AMB TEST SET
%% ============================================================================

fprintf('=== AVALUACIÓ AMB TEST SET ===\n');

% Extreure característiques del test
fprintf('Extracting features from TEST set...\n');
[XTest, testValid] = extractFeatures(imdsTest, config);
YTest = imdsTest.Labels(testValid);

% Aplicar la mateixa normalització que al train (amb protecció)
XTest_norm = (XTest - mu) ./ sigma;

% Reemplaçar qualsevol NaN/Inf que pugui aparèixer amb 0
XTest_norm(~isfinite(XTest_norm)) = 0;

fprintf('Test features: %d imatges vàlides\n\n', size(XTest_norm, 1));

% Verificar que tenim dades vàlides
if size(XTest_norm, 1) == 0
    error('No hi ha imatges vàlides al test set! Revisa l''extracció de característiques.');
end

% Prediccions
YPred_svm = predict(mdl_svm, XTest_norm);
YPred_rf = categorical(predict(mdl_rf, XTest_norm));
YPred_knn = predict(mdl_knn, XTest_norm);

% Càlcul d'accuracy
acc_svm = sum(YPred_svm == YTest) / numel(YTest) * 100;
acc_rf = sum(YPred_rf == YTest) / numel(YTest) * 100;
acc_knn = sum(YPred_knn == YTest) / numel(YTest) * 100;

fprintf('RESULTATS FINALS:\n');
fprintf('  SVM:           %.2f%%\n', acc_svm);
fprintf('  Random Forest: %.2f%%\n', acc_rf);
fprintf('  KNN:           %.2f%%\n\n', acc_knn);

% Seleccionar millor model
[bestAcc, bestIdx] = max([acc_svm, acc_rf, acc_knn]);
modelNames = {'SVM', 'Random Forest', 'KNN'};
predictions = {YPred_svm, YPred_rf, YPred_knn};

fprintf('MILLOR MODEL: %s (%.2f%%)\n\n', modelNames{bestIdx}, bestAcc);

%% ============================================================================
%% 6. VISUALITZACIÓ DE RESULTATS
%% ============================================================================

% Matriu de confusió
figure('Position', [100 100 800 700]);
cm = confusionchart(YTest, predictions{bestIdx});
cm.Title = sprintf('Matriu de Confusió - %s', modelNames{bestIdx});
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Mètriques per classe
figure('Position', [100 100 1200 400]);
classes = categories(YTest);
precision = zeros(length(classes), 1);
recall = zeros(length(classes), 1);
f1score = zeros(length(classes), 1);

for i = 1:length(classes)
    tp = sum(predictions{bestIdx} == classes{i} & YTest == classes{i});
    fp = sum(predictions{bestIdx} == classes{i} & YTest ~= classes{i});
    fn = sum(predictions{bestIdx} ~= classes{i} & YTest == classes{i});
    
    precision(i) = tp / (tp + fp + eps);
    recall(i) = tp / (tp + fn + eps);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

subplot(1,3,1); bar(precision); title('Precision per Classe'); 
set(gca, 'XTickLabel', classes, 'XTickLabelRotation', 45);
ylabel('Precision'); ylim([0 1]);

subplot(1,3,2); bar(recall); title('Recall per Classe');
set(gca, 'XTickLabel', classes, 'XTickLabelRotation', 45);
ylabel('Recall'); ylim([0 1]);

subplot(1,3,3); bar(f1score); title('F1-Score per Classe');
set(gca, 'XTickLabel', classes, 'XTickLabelRotation', 45);
ylabel('F1-Score'); ylim([0 1]);

% Mostrar exemples de prediccions
figure('Position', [100 100 1200 800]);
nExamples = min(12, numel(YTest));
rng(42);
idxs = randperm(numel(YTest), nExamples);

% Trobar els índexs originals de les imatges vàlides
validIndices = find(testValid);

for i = 1:nExamples
    subplot(3, 4, i);
    
    % Agafar l'índex original de la imatge al imageDatastore
    originalIdx = validIndices(idxs(i));
    img = readimage(imdsTest, originalIdx);
    imshow(imresize(img, [128 128]));
    
    actual = char(YTest(idxs(i)));
    predicted = char(predictions{bestIdx}(idxs(i)));
    isCorrect = strcmp(actual, predicted);
    
    if isCorrect
        titleColor = [0 0.7 0];
        titleStr = sprintf('✓ %s', actual);
    else
        titleColor = [0.8 0 0];
        titleStr = sprintf('✗ Real: %s\nPred: %s', actual, predicted);
    end
    
    title(titleStr, 'Color', titleColor, 'FontSize', 8);
end
sgtitle(sprintf('Exemples de Prediccions - %s (%.2f%%)', modelNames{bestIdx}, bestAcc));

%% ============================================================================
%% FUNCIONS AUXILIARS
%% ============================================================================

function [features, validMask] = extractFeatures(imds, config)
    % Extreu característiques de totes les imatges d'un imageDatastore
    
    numImages = numel(imds.Files);
    numFeatures = config.fourierDesc + config.huMoments + ...
                  config.colorBins * 3 + config.shapeFeats;
    
    features = zeros(numImages, numFeatures);
    validMask = true(numImages, 1);
    
    for i = 1:numImages
        if mod(i, 100) == 0
            fprintf('  Processant %d/%d...\n', i, numImages);
        end
        
        try
            img = readimage(imds, i);
            feat = extractSingleFeature(img, config);
            
            % Validar que no hi hagi NaN/Inf
            if any(isnan(feat)) || any(isinf(feat))
                validMask(i) = false;
            else
                features(i, :) = feat;
            end
        catch
            validMask(i) = false;
        end
    end
end

function feat = extractSingleFeature(img, config)
    % Extreu característiques d'una sola imatge
    
    % Redimensionar
    img = imresize(img, [config.imgSize config.imgSize]);
    
    % Preprocessament
    img_gray = rgb2gray(img);
    
    % Detecció de vores millorada
    bw = edge(img_gray, 'canny', [0.1 0.3]);
    se = strel('disk', 2);
    bw = imclose(bw, se);
    bw = imfill(bw, 'holes');
    bw = bwareaopen(bw, 30);
    
    % Inicialitzar vector de característiques
    feat = zeros(1, config.fourierDesc + config.huMoments + ...
                 config.colorBins * 3 + config.shapeFeats);
    idx = 1;
    
    %% 1. DESCRIPTORS DE FOURIER
    boundaries = bwboundaries(bw);
    if ~isempty(boundaries)
        [~, maxIdx] = max(cellfun(@length, boundaries));
        boundary = boundaries{maxIdx};
        
        if size(boundary, 1) >= config.fourierDesc
            s = boundary(:,2) + 1i * boundary(:,1);
            z = fft(s);
            z_mag = abs(z);
            z_norm = z_mag(2:end);
            
            if ~isempty(z_norm) && z_norm(1) > 0
                z_norm = z_norm / z_norm(1);
                feat(idx:idx+config.fourierDesc-1) = z_norm(1:config.fourierDesc);
            end
        end
    end
    idx = idx + config.fourierDesc;
    
    %% 2. MOMENTS DE HU
    hu = computeHuMoments(bw);
    feat(idx:idx+config.huMoments-1) = hu;
    idx = idx + config.huMoments;
    
    %% 3. HISTOGRAMES DE COLOR (HSV - millor per senyals)
    img_hsv = rgb2hsv(img);
    for ch = 1:3
        channel = img_hsv(:,:,ch);
        hist_vals = histcounts(channel(:), config.colorBins, ...
                              'BinLimits', [0 1], ...
                              'Normalization', 'probability');
        feat(idx:idx+config.colorBins-1) = hist_vals;
        idx = idx + config.colorBins;
    end
    
    %% 4. CARACTERÍSTIQUES DE FORMA
    props = regionprops(bw, 'Area', 'Perimeter', 'Eccentricity', ...
                        'Solidity', 'Extent', 'MajorAxisLength');
    if ~isempty(props)
        p = props(1);
        feat(idx) = p.Area / (config.imgSize^2);
        feat(idx+1) = p.Eccentricity;
        feat(idx+2) = p.Solidity;
        feat(idx+3) = p.Extent;
        if p.Perimeter > 0
            feat(idx+4) = 4*pi*p.Area / (p.Perimeter^2); % Circularitat
        end
        feat(idx+5) = p.MajorAxisLength / config.imgSize;
    end
end

function hu = computeHuMoments(bw)
    % Calcula els 7 moments invariants de Hu
    
    hu = zeros(1, 7);
    
    [y, x] = find(bw);
    if isempty(x)
        return;
    end
    
    % Centroide
    xc = mean(x);
    yc = mean(y);
    
    % Moments centrals
    mu20 = sum((x - xc).^2);
    mu02 = sum((y - yc).^2);
    mu11 = sum((x - xc).*(y - yc));
    mu30 = sum((x - xc).^3);
    mu03 = sum((y - yc).^3);
    mu21 = sum((x - xc).^2.*(y - yc));
    mu12 = sum((x - xc).*(y - yc).^2);
    
    % Normalitzar
    n = length(x);
    nu20 = mu20 / n^2;
    nu02 = mu02 / n^2;
    nu11 = mu11 / n^2;
    nu30 = mu30 / n^2.5;
    nu03 = mu03 / n^2.5;
    nu21 = mu21 / n^2.5;
    nu12 = mu12 / n^2.5;
    
    % Calcular moments de Hu
    hu(1) = nu20 + nu02;
    hu(2) = (nu20 - nu02)^2 + 4*nu11^2;
    hu(3) = (nu30 - 3*nu12)^2 + (3*nu21 - nu03)^2;
    hu(4) = (nu30 + nu12)^2 + (nu21 + nu03)^2;
    hu(5) = (nu30 - 3*nu12)*(nu30 + nu12)*((nu30 + nu12)^2 - 3*(nu21 + nu03)^2) + ...
            (3*nu21 - nu03)*(nu21 + nu03)*(3*(nu30 + nu12)^2 - (nu21 + nu03)^2);
    hu(6) = (nu20 - nu02)*((nu30 + nu12)^2 - (nu21 + nu03)^2) + ...
            4*nu11*(nu30 + nu12)*(nu21 + nu03);
    hu(7) = (3*nu21 - nu03)*(nu30 + nu12)*((nu30 + nu12)^2 - 3*(nu21 + nu03)^2) - ...
            (nu30 - 3*nu12)*(nu21 + nu03)*(3*(nu30 + nu12)^2 - (nu21 + nu03)^2);
    
    % Log per estabilitat
    hu = -sign(hu) .* log10(abs(hu) + 1e-10);
end