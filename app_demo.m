% APP DE DEMOSTRACIÓ - DETECCIÓ DE SENYALS
% Carrega el model entrenat i prediu una imatge nova

clear; clc; close all;
addpath('src'); % Necessitem accés a extractSingleFeature

%% 1. CARREGAR EL CERVELL (MODEL)
if ~isfile('model_entrenat.mat')
    error('No trobo el fitxer "model_entrenat.mat". Executa primer el main.m!');
end

fprintf('Carregant model...\n');
load('model_entrenat.mat'); % Carrega: finalModel, config, trainMu, trainSigma, bestModelName

%% 2. SELECCIONAR IMATGE
[file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.ppm', 'Imatges de Senyals'}, ...
                         'Selecciona una senyal de trànsit');

if isequal(file, 0)
    disp('Operació cancel·lada per l''usuari.');
    return;
end

fullPath = fullfile(path, file);
img = imread(fullPath);

%% 3. EXTRACCIÓ I PREDICCIÓ
try
    % A. Extreure característiques (igual que al training)
    feat = extractSingleFeature(img, config);
    
    % B. Normalitzar (USANT LA MU/SIGMA DEL TRAINING!)
    % Això és crucial: hem d'escalar les dades igual que com va aprendre el model
    feat_norm = (feat - trainMu) ./ trainSigma;
    
    % C. Gestionar NaNs si n'hi ha (casos rars)
    feat_norm(~isfinite(feat_norm)) = 0;
    
    % D. Predir
    % La sintaxi pot variar lleugerament entre SVM/KNN i Random Forest
    if contains(bestModelName, 'Random Forest')
        prediction = predict(finalModel, feat_norm);
        label = prediction{1}; % RF torna un cell array
    else
        label = predict(finalModel, feat_norm); % SVM/KNN tornen categorical/string
        label = char(label);
    end
    
    %% 4. MOSTRAR RESULTAT
    figure('Name', 'Resultat de la Predicció', 'NumberTitle', 'off', 'Color', 'w');
    
    % Mostrar imatge
    imshow(img);
    
    % Mostrar títol bonic
    title({sprintf('MODEL: %s', bestModelName), ...
           sprintf('PREDICCIÓ: \fontsize{16}\color{blue}%s', label)}, ...
           'Interpreter', 'tex');
       
    fprintf('Imatge: %s\nPredicció: %s\n', file, label);
    
catch ME
    errordlg(['Error processant la imatge: ' ME.message], 'Error');
end