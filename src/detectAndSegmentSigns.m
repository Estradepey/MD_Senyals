function [candidates, bboxes] = detectAndSegmentSigns(img, config)
% DETECTANDSEGMENTSIGNS Detecta per FORMA i valida per COLOR
% 1. Detecta cercles i polígons.
% 2. Retalla la regió candidata.
% 3. Comprova si dins del retall hi ha prou color Vermell o Blau.

    candidates = {};
    bboxes = [];
    
    if size(img, 3) == 3
        imgGray = rgb2gray(img);
    else
        imgGray = img;
    end
    
    [imgH, imgW] = size(imgGray);
    minDim = min(imgH, imgW);
    
    %% 1. DETECCIÓ DE FORMES (CERCLES - HOUGH)
    minRadius = 15; 
    maxRadius = floor(minDim / 2 * 1.1); % Permet senyals gegants

    % Busquem cercles
    [centers, radii] = imfindcircles(img, [minRadius maxRadius], ...
                                     'ObjectPolarity', 'dark', ... 
                                     'Sensitivity', 0.92, 'Method', 'TwoStage');
    
    % Per a senyals brillants (ex: fons blanc vora fosca)
    [centersB, radiiB] = imfindcircles(img, [minRadius maxRadius], ...
                                     'ObjectPolarity', 'bright', ...
                                     'Sensitivity', 0.92);
    
    centers = [centers; centersB];
    radii = [radii; radiiB];
    
    % Eliminem cercles concèntrics (mateixa lògica que abans)
    if ~isempty(radii)
        [radii, sortIdx] = sort(radii, 'descend');
        centers = centers(sortIdx, :);
        keepIdx = true(size(radii));
        
        for i = 1:length(radii)
            if ~keepIdx(i), continue; end
            for j = (i+1):length(radii)
                if ~keepIdx(j), continue; end
                distCenters = norm(centers(i,:) - centers(j,:));
                if distCenters < (radii(i) * 0.3) % Si està dins del 30% del radi
                    keepIdx(j) = false; % Eliminem el petit
                end
            end
        end
        radii = radii(keepIdx);
        centers = centers(keepIdx, :);
        
        % PROCESSAR CERCLES
        for i = 1:length(radii)
            r = radii(i);
            c = centers(i, :);
            
            x1 = round(c(1) - r);
            y1 = round(c(2) - r);
            w = round(2*r); h = round(2*r);
            
            [crop, box] = safeCrop(img, [x1, y1, w, h], config);
            
            % --- AQUI AFEGIM EL FILTRE DE COLOR ---
            if ~isempty(crop) && checkColorPresence(crop)
                candidates{end+1} = crop; %#ok<AGROW>
                bboxes = [bboxes; box];   %#ok<AGROW>
            end
        end
    end

    %% 2. DETECCIÓ DE FORMES (TRIANGLES/OCTÀGONS - MORFOLOGIA)
    imgDenoised = medfilt2(imgGray, [3 3]);
    bwEdges = edge(imgDenoised, 'canny', [0.05 0.2]);
    se = strel('disk', 2);
    bwDilated = imdilate(bwEdges, se);
    bwFilled = imfill(bwDilated, 'holes');
    bwClean = bwareaopen(bwFilled, 150);
    
    stats = regionprops(bwClean, 'BoundingBox', 'Area', 'Perimeter', 'Solidity', 'Extent', 'Centroid');
    
    for k = 1:length(stats)
        area = stats(k).Area;
        box = stats(k).BoundingBox; 
        
        w = box(3); h = box(4);
        aspectRatio = w / h;
        
        if area < 400 || aspectRatio < 0.5 || aspectRatio > 2.0
            continue;
        end
        
        extent = stats(k).Extent;
        solidity = stats(k).Solidity;
        circularity = (4 * pi * area) / (stats(k).Perimeter^2);
        
        isShapeOfInterest = false;
        
        % Triangle (Cediu el pas / Perill)
        if (extent > 0.4 && extent < 0.65) && (solidity > 0.8)
             isShapeOfInterest = true;
        end
        
        % Octàgon (STOP)
        if (circularity > 0.85) && (solidity > 0.9) && (extent > 0.75)
            isShapeOfInterest = true;
        end
        
        if isShapeOfInterest
             % Check duplicats contra bboxes ja existents
             polyCenter = stats(k).Centroid;
             if ~isCloseToExisting(polyCenter, bboxes)
                [crop, finalBox] = safeCrop(img, box, config);
                
                % --- AQUI AFEGIM EL FILTRE DE COLOR ---
                if ~isempty(crop) && checkColorPresence(crop)
                    candidates{end+1} = crop; %#ok<AGROW>
                    bboxes = [bboxes; finalBox]; %#ok<AGROW>
                end
             end
        end
    end
% Opcional: Visualització per debug
     figure, imshow(img), hold on
     for i=1:size(bboxes,1), rectangle('Position',bboxes(i,:),'EdgeColor','g','LineWidth',2); end


end

%% --- FUNCIÓ NOVA: FILTRE DE COLOR ---
function pass = checkColorPresence(cropImg)
    % Convertim el retall (crop) a HSV
    hsv = rgb2hsv(cropImg);
    h = hsv(:,:,1);
    s = hsv(:,:,2);
    v = hsv(:,:,3);
    
    % Definir llindars per als colors dels senyals de trànsit
    % Necessitem alta saturació per evitar detectar grisos/blancs com a color
    
    % 1. VERMELL (Té dos pics al rang Hue: principi i final)
    maskRed1 = (h > 0.92) & (s > 0.45) & (v > 0.2);
    maskRed2 = (h < 0.08) & (s > 0.45) & (v > 0.2);
    maskRed = maskRed1 | maskRed2;
    
    % 2. BLAU (Senyals obligació / vianants / autopista)
    maskBlue = (h > 0.55 & h < 0.70) & (s > 0.45) & (v > 0.2);
    
    % 3. Opcional: GROC (Obres) - Descomenta si cal
    maskYellow = (h > 0.12 & h < 0.18) & (s > 0.5) & (v > 0.3);
    
    % Unim màscares (només ens interessa si té algun d'aquests colors)
    maskTotal = maskRed | maskBlue | maskYellow;
    
    % Calculem el percentatge de píxels "de senyal" en el retall
    totalPixels = numel(h);
    signPixels = sum(maskTotal(:));
    ratio = signPixels / totalPixels;
    
    % LLINDAR DE VALIDACIÓ:
    % Si almenys el 8% de la imatge és del color correcte, l'acceptem.
    % Un cercle vermell (prohibició) té molt blanc al mig, així que 
    % el ratio de vermell pot ser baix (per això posem 0.08 o 8%).
    pass = ratio > 0.08; 
end

% --- FUNCIONS AUXILIARS EXISTENTS ---

function isClose = isCloseToExisting(center, existingBoxes)
    isClose = false;
    if isempty(existingBoxes), return; end
    for i = 1:size(existingBoxes, 1)
        b = existingBoxes(i, :);
        cExists = [b(1)+b(3)/2, b(2)+b(4)/2];
        radiusExists = max(b(3), b(4)) / 2;
        if norm(center - cExists) < (radiusExists * 0.5)
            isClose = true; return;
        end
    end
end

function [cropResized, realBox] = safeCrop(img, box, config)
    [imgH, imgW, ~] = size(img);
    pad = round(max(box(3), box(4)) * 0.1); 
    
    x1 = floor(max(1, box(1) - pad));
    y1 = floor(max(1, box(2) - pad));
    x2 = ceil(min(imgW, box(1) + box(3) + pad));
    y2 = ceil(min(imgH, box(2) + box(4) + pad));
    
    w = x2 - x1 + 1; h = y2 - y1 + 1;
    crop = img(y1:y2, x1:x2, :);
    realBox = [x1, y1, w, h];
    
    if isempty(crop), cropResized = []; return; end

    if isfield(config, 'imageSize')
        targetSize = config.imageSize;
    else
        targetSize = [32 32];
    end
    cropResized = imresize(crop, targetSize(1:2));
end