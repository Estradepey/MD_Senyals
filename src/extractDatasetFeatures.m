function [X_norm, Y, mu, sigma] = extractDatasetFeatures(imds, config, mu, sigma)
    % Versió amb DEPURACIÓ D'ERRORS
    
    numImages = numel(imds.Files);
    
    % Fem una prova amb la primera imatge per calcular la mida real de les features
    try
        imgTest = readimage(imds, 1);
        featTest = extractSingleFeature(imgTest, config);
        numFeats = length(featTest);
        fprintf('  Mida del vector de features: %d\n', numFeats);
    catch ME
        error('ERROR CRÍTIC PROVANT LA PRIMERA IMATGE: %s\nRevisa extractSingleFeature.m', ME.message);
    end
    
    X_raw = zeros(numImages, numFeats);
    validMask = true(numImages, 1);
    
    % Bucle d'extracció
    % Canviem 'parfor' a 'for' normal per veure millor els errors mentre depurem
    for i = 1:numImages 
        try
            img = readimage(imds, i);
            feat = extractSingleFeature(img, config);
            
            % Comprovació de consistència
            if length(feat) ~= numFeats
                error('Mida incorrecta de features. Esperat: %d, Trobat: %d', numFeats, length(feat));
            end
            
            if any(isnan(feat)) || any(isinf(feat))
                fprintf(2, 'AVÍS: Imatge %d conté NaNs o Infs.\n', i);
                validMask(i) = false;
            else
                X_raw(i, :) = feat;
            end
        catch ME
            % AQUÍ ÉS ON MOSTREM L'ERROR REAL
            fprintf(2, 'ERROR a la imatge %d: %s\n', i, ME.message);
            validMask(i) = false;
            
            % Si fallen les 3 primeres, parem perquè no s'ompli la pantalla
            if i <= 3
                rethrow(ME); 
            end
        end
    end
    
    % Filtrar invàlids
    X_raw = X_raw(validMask, :);
    Y = imds.Labels(validMask);
    
    if isempty(X_raw)
        return; % Tornem buit, l'error saltarà a trainModels
    end
    
    % Càlcul de normalització
    if isempty(mu) || isempty(sigma)
        mu = mean(X_raw, 1);
        sigma = std(X_raw, 0, 1);
        sigma(sigma < 1e-10) = 1; 
    end
    
    % Normalitzar
    X_norm = (X_raw - mu) ./ sigma;
    
    % Neteja final
    validFinal = all(isfinite(X_norm), 2);
    X_norm = X_norm(validFinal, :);
    Y = Y(validFinal);
    
    fprintf('  -> Final: %d mostres vàlides de %d originals.\n', size(X_norm, 1), numImages);
end