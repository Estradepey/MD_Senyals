% APP DE DEMOSTRACIÓ - DETECCIÓ DE SENYALS EN ESCENA COMPLETA
% 1. Detecta zones d'interès (segmentació + patching)
% 2. Classifica cada zona (model ML)

clear; clc; close all;
addpath('src'); 

warning('off', 'stats:pdist2:DataConversion');

%% 1. CARREGAR MODEL
if ~isfile('model_entrenat.mat')
    error('No trobo "model_entrenat.mat". Executa primer main.m!');
end
fprintf('Carregant model...\n');
load('model_entrenat.mat'); % Variables: finalModel, config, trainMu, trainSigma, bestModelName

%% 2. SELECCIONAR IMATGE (ESCENA COMPLETA)
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Imatges de Carrer'}, ...
                         'Selecciona una foto de carrer');
if isequal(file, 0), return; end

fullPath = fullfile(path, file);
img = imread(fullPath);

%% 3. FASE 1: Detecció i Patching
fprintf('Escanejant la imatge...\n');

[candidates, bboxes] = detectAndSegmentSigns(img, config);

numCandidates = length(candidates);
fprintf('S''han trobat %d possibles senyals.\n', numCandidates);

%% 4. FASE 2: Classificació

% Preparem la visualització
figure('Name', 'Resultat Final', 'Color', 'w');
imshow(img); hold on;
title(sprintf('Detecció amb %s', bestModelName));

if numCandidates == 0
    text(10, 20, 'CAP SENYAL TROBADA', 'Color', 'r', 'FontSize', 14, 'FontWeight', 'bold');
end

% Bucle per analitzar cada "patch" trobat
for i = 1:numCandidates
    patch = candidates{i};
    box = bboxes(i, :); % Coordenades [x y w h]
    
    try
        % A. Extreure característiques
        feat = extractSingleFeature(patch, config);
        
        % B. Normalitzar (IMP: Usar mu/sigma del training)
        feat_norm = (feat - trainMu) ./ trainSigma;
        feat_norm = single(feat_norm);        feat_norm(~isfinite(feat_norm)) = 0; % Seguretat
        
% C. Predir con SCORES (Probabilidades)
        [prediction, scores] = predict(finalModel, feat_norm);
        
        % Ordenar las puntuaciones de mayor a menor
        sortedScores = sort(scores, 'descend');
        topScore = sortedScores(1);
        secondScore = sortedScores(2);
        
        % CRITERIO DE CONFIANZA:
        % 1. Que la probabilidad sea decente 
        % 2. Que haya diferencia con la segunda opción 

        if topScore < 0.4 || (topScore - secondScore) < 0.2
            fprintf('  -> Objecte %d: NO CLASSIFICAT (Confiança: %.2f%%)\n', i, topScore*100);
            rectangle('Position', box, 'EdgeColor', 'g', 'LineWidth', 3);
            continue; % Salta a la següent iteració del bucle
        end

        
        % Si pasa el filtro, dibujamos...
        label = string(prediction);
        
        if label == "fons" || label == "background" 
            fprintf('  -> Objecte %d: Ignorat (Classificat com fons)\n', i);
            continue; % Salta a la siguiente iteración del bucle
        end

        % D. DIBUIXAR RESULTAT
        % Dibuixem el rectangle verd
        rectangle('Position', box, 'EdgeColor', 'g', 'LineWidth', 3);
        
        % Posem l'etiqueta a sobre
        text(box(1), box(2)-10, char(label), ...
             'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold', ...
             'BackgroundColor', 'black');
         
fprintf('  -> Objecte %d: Classificat com "%s" (Confiança: %.2f%%)\n', ...
                i, label, topScore*100);        
    catch ME
        warning('Error processant candidat %d: %s', i, ME.message);
    end
end
hold off;