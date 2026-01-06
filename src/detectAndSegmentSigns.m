function [candidates, bboxes] = detectAndSegmentSigns(img, config)
% DETECTANDSEGMENTSIGNS Detecta senyals usant Color (HSV) i Morfologia
% Aquesta funció substitueix l'enfocament anterior per un més robust
% per a càmeres de mòbil.

    % 1. Pre-processament: Convertir a HSV
    hsv = rgb2hsv(img);
    h = hsv(:,:,1); 
    s = hsv(:,:,2); 
    v = hsv(:,:,3);
    
    % 2. Màscares de Color (Vermell i Blau)
    % Vermell (té un problema, està al principi i al final de l'espectre H)
    maskRed1 = (h > 0.94) & (s > 0.45) & (v > 0.2);
    maskRed2 = (h < 0.06) & (s > 0.45) & (v > 0.2);
    maskRed = maskRed1 | maskRed2;
    
    % Blau (Senyals d'obligació)
    maskBlue = (h > 0.5 & h < 0.75) & (s > 0.4) & (v > 0.25);

    % Groc (senyals de perill / vianants)
    % El groc està aproximadament al voltant de H ≈ 0.16 (60º)
    % Deixem un rang una mica ample i demanem saturació i brillantor altes
    maskYellow = (h > 0.10 & h < 0.20) & (s > 0.4) & (v > 0.4);
    
    % Unim màscares
    maskTotal = maskRed | maskBlue | maskYellow;
    
    % 3. Neteja Morfològica
    % Eliminar soroll petit
    maskTotal = bwareaopen(maskTotal, 100); 
    % Tancar forats i suavitzar formes
    se = strel('disk', 1);
    maskTotal = imclose(maskTotal, se);
    maskTotal = imfill(maskTotal, 'holes');
    
    % 4. Extracció de regions (Regionprops)
    stats = regionprops(maskTotal, 'BoundingBox', 'Area', 'Eccentricity', 'Solidity');
    
    candidates = {};
    bboxes = [];
    
    [imgH, imgW, ~] = size(img);
    
    for k = 1:length(stats)
        box = stats(k).BoundingBox; % [x y w h]
        area = stats(k).Area;
        eccentricity = stats(k).Eccentricity; 
        solidity = stats(k).Solidity;
        
        w = box(3);
        h = box(4);
        aspectRatio = w / h;
        
        % --- FILTRES GEOMÈTRICS ---
        % Mida mínima i màxima
        if area < 400 || area > (imgH * imgW * 0.6)
            continue;
        end
        
        % Proporció (Aspect Ratio) - Els senyals són quadrats/cercles (aprox 1)
        if aspectRatio < 0.5 || aspectRatio > 2.0
            continue;
        end
        
        % Solidesa - Els senyals són sòlids, no formes estranyes
        if solidity < 0.45
            continue;
        end
        
        % 5. Retall i Padding (Marge de seguretat)
        padding = round(max(w, h) * 0.1); 
        x1 = floor(max(1, box(1) - padding));
        y1 = floor(max(1, box(2) - padding));
        x2 = ceil(min(imgW, box(1) + w + padding));
        y2 = ceil(min(imgH, box(2) + h + padding));
        
        % Retallem la imatge original
        crop = img(y1:y2, x1:x2, :);
        
        % Redimensionem per al model
        if isfield(config, 'imageSize')
            targetSize = config.imageSize;
        else
            targetSize = [64 64];
        end
        
        if ~isempty(crop)
            cropResized = imresize(crop, targetSize(1:2));
            candidates{end+1} = cropResized; %#ok<AGROW>
            bboxes = [bboxes; [x1, y1, (x2-x1+1), (y2-y1+1)]]; %#ok<AGROW>
        end
    end
end