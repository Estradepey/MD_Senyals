%% ---------------------------------------------------------
%  PRÀCTICA RECONEIXEMENT DE SENYALS DE TRÀNSIT
%  Script Principal - Entrenament i Avaluació
% ----------------------------------------------------------

clc; clear; close all;

%% 1. CONFIGURACIÓ I CÀRREGA DE DADES
% ==========================================================
% Canvia aquesta ruta per la del teu ordinador:
datasetPath = 'C:\Users\max.estrade\Downloads\imatges_senyals'; 

disp('Carregant imatges...');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Separació: 70% Entrenament, 30% Test
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

disp(['Imatges Train: ', num2str(numel(imdsTrain.Files))]);
disp(['Imatges Test:  ', num2str(numel(imdsTest.Files))]);

%% 2. EXTRACCIÓ DE CARACTERÍSTIQUES (ENTRENAMENT)
% ==========================================================
disp('Extraient característiques del conjunt d''entrenament...');

% Primer, llegim una imatge per saber la mida del vector de característiques
imgExemple = readimage(imdsTrain, 1);
featExemple = getTrafficSignFeatures(imgExemple);
numFeat = length(featExemple);

% Pre-assignem memòria per velocitat
XTrain = zeros(numel(imdsTrain.Files), numFeat);
YTrain = imdsTrain.Labels;

% Bucle d'extracció
for i = 1:numel(imdsTrain.Files)
    img = readimage(imdsTrain, i);
    
    % Cridem a la funció personalitzada (veure al final del script)
    XTrain(i, :) = getTrafficSignFeatures(img);
    
    % Barra de progrés simple
    if mod(i, 100) == 0
        fprintf('Processades %d / %d\n', i, numel(imdsTrain.Files));
    end
end

%% 3. ENTRENAMENT DEL MODEL (SVM Multiclasse)
% ==========================================================
disp('Entrenant el classificador SVM (fitcecoc)...');

% CORRECCIÓ: Traiem 'ClassWeights' que donava error a la teva versió.
% La millora vindrà donada pels millors descriptors (HOG 4x4), no per això.
t = templateSVM('Standardize', true, 'KernelFunction', 'gaussian');

% Si vols equilibrar classes (perquè tens pocs STOPs), 
% es fa aquí al fitcecoc amb 'Prior', 'uniform'.
% Això dóna la mateixa importància a totes les classes.
model = fitcecoc(XTrain, YTrain, 'Learners', t, 'Prior', 'uniform');

disp('Entrenament finalitzat.');

%% 4. AVALUACIÓ AMB EL CONJUNT DE TEST
% ==========================================================
disp('Avaluant el model amb imatges de test...');

XTest = zeros(numel(imdsTest.Files), numFeat);
YTest = imdsTest.Labels;

% Bucle d'extracció per Test
for i = 1:numel(imdsTest.Files)
    img = readimage(imdsTest, i);
    XTest(i, :) = getTrafficSignFeatures(img);
end

% Predicció
YPred = predict(model, XTest);

% Càlcul de precisió (Accuracy)
accuracy = mean(YPred == YTest);
fprintf('\n--------------------------------------------\n');
fprintf('PRECISIÓ FINAL (ACCURACY): %.2f%%\n', accuracy * 100);
fprintf('--------------------------------------------\n');

% Matriu de Confusió Visual
figure;
confusionchart(YTest, YPred);
title(['Matriu de Confusió - Accuracy: ' num2str(accuracy*100) '%']);


%% ---------------------------------------------------------
%  FUNCIÓ LOCAL: EXTRACCIÓ DE CARACTERÍSTIQUES
%  Aquesta funció és el "cervell" del sistema.
% ----------------------------------------------------------
function features = getTrafficSignFeatures(img)
    
    % 0. CORRECCIÓ FORMAT
    if isa(img, 'uint16'), img = im2uint8(img); end
    if ~isa(img, 'uint8') && ~isa(img, 'double'), img = im2uint8(img); end

    % 1. REDIMENSIONAT
    imgSize = [64, 64]; 
    img = imresize(img, imgSize);
    
    % ------------------------------------------------------
    % A. HOG (Forma detallada)
    % ------------------------------------------------------
    if size(img, 3) == 3
        imgGray = rgb2gray(img);
    else
        imgGray = img;
    end
    
    [hogFeat, ~] = extractHOGFeatures(imgGray, 'CellSize', [4 4]);
    
    % ------------------------------------------------------
    % B. COLOR (HSV + RGB)
    % ------------------------------------------------------
    if size(img, 3) == 3
        imgHSV = rgb2hsv(img);
        meanR = mean(mean(img(:,:,1)));
        meanG = mean(mean(img(:,:,2)));
        meanB = mean(mean(img(:,:,3)));
        avgSat = mean(mean(imgHSV(:,:,2)));
        colorFeat = [meanR, meanG, meanB, avgSat];
    else
        colorFeat = [0, 0, 0, 0];
    end
    
    % ------------------------------------------------------
    % C. GEOMETRIA (Corregida per evitar errors)
    % ------------------------------------------------------
    bw = imbinarize(imgGray);
    bw = imfill(bw, 'holes'); % Omplir forats (important per l'STOP)
    bw = bwareafilt(bw, 1);   % Quedar-se amb l'objecte més gran
    
    if ~any(bw(:))
        propsVector = [0, 0, 0, 0];
    else
        % CANVI AQUÍ: Demanem 'Area' i 'Perimeter' en lloc de 'Circularities'
        props = regionprops(bw, 'Solidity', 'Extent', 'Eccentricity', 'Area', 'Perimeter');
        
        % Càlcul manual de la circularitat (Funciona en totes les versions)
        % C = (4 * pi * Area) / Perímetre^2
        % Un cercle perfecte dóna 1. Un quadrat o octàgon dóna menys.
        area = props(1).Area;
        perim = props(1).Perimeter;
        
        if perim == 0
            circ = 0;
        else
            circ = (4 * pi * area) / (perim^2);
        end
        
        propsVector = [props(1).Solidity, props(1).Extent, props(1).Eccentricity, circ];
    end
    
    % ------------------------------------------------------
    % CONCATENACIÓ
    % ------------------------------------------------------
    features = [hogFeat, colorFeat, propsVector];
    
end