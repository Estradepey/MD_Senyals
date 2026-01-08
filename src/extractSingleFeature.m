function feat = extractSingleFeature(img, config)
    %% 0. PRE-PROCESSAMENT
    % Si el patching falla
    if isempty(img)
        feat = []; return;
    end

    % Si la imatge ve en format 16 bits, la passem a 8 bits
    if isa(img, 'uint16')
        img = im2uint8(img);
    end

    % Intentar segmentar el senyal dins de la imatge
    try
        [candidates, ~] = detectAndSegmentSigns(img, config);
        if ~isempty(candidates)
            % Fem servir el primer candidat (patch amb la senyal)
            img = candidates{1};
        end
    catch
        % Si la segmentació falla per qualsevol motiu, continuem amb la
        % imatge original sense modificar-la.
    end

    %(MIDA FIXA)
    img = imresize(img, [config.imgSize config.imgSize]);
    
    img_gray = rgb2gray(img);
    
    %(Vores)
    bw = edge(img_gray, 'canny', [0.1 0.3]);
    bw= imclose(bw, strel('disk', 2));
    bw = imfill(bw, 'holes');
    bw = bwareaopen(bw, 30);
    
    feat = [];
    
    %% 1. FOURIER
    fourierFeat = zeros(1, config.fourierDesc);
    boundaries = bwboundaries(bw);
    if ~isempty(boundaries)
        [~, maxIdx] = max(cellfun(@length, boundaries));
        boundary = boundaries{maxIdx};
        if size(boundary, 1) >= config.fourierDesc
            s = boundary(:,2) + 1i * boundary(:,1);
            z = abs(fft(s));
            z = z(2:end); % Treure component DC
            if z(1) > 0, z = z / z(1); end % Normalitzar escala
            fourierFeat = z(1:config.fourierDesc).';
        end
    end
    feat = [feat, fourierFeat];
    
    %% 2. COLOR (HSV)
    img_hsv = rgb2hsv(img);
    colorFeat = [];
    for ch = 1:3
        h = histcounts(img_hsv(:,:,ch), config.colorBins, ...
                       'BinLimits', [0 1], 'Normalization', 'probability');
        colorFeat = [colorFeat, h];
    end
    feat = [feat, colorFeat];
    
    %% 3. FORMA
    props = regionprops(bw, 'Area', 'Perimeter', 'Eccentricity', ...
                        'Solidity', 'Extent', 'MajorAxisLength');
    shapeFeat = zeros(1, 6);
    if ~isempty(props)
        p = props(1);
        shapeFeat(1) = p.Area / (config.imgSize^2);
        shapeFeat(2) = p.Eccentricity;
        shapeFeat(3) = p.Solidity;
        shapeFeat(4) = p.Extent;
        if p.Perimeter > 0
            shapeFeat(5) = 4*pi*p.Area / (p.Perimeter^2);
        end
        shapeFeat(6) = p.MajorAxisLength / config.imgSize;
    end
    feat = [feat, shapeFeat];

    %% 4. HOG (Histogram of Oriented Gradients)
    
    % 'CellSize', Quant mes petit dona detalls fins.
    hogFeat = extractHOGFeatures(img_gray, 'CellSize', [config.cellSize config.cellSize]);
    
    % Afegir al vector final
    feat = [feat, hogFeat];

    %% 5. SIFT (descriptors de textura local) - Reduït i normalitzat
    siftDim = config.siftDim;  % Reduïm dimensionalitat per evitar que dominin
    siftFeat = zeros(1, siftDim);

    try
        points = detectSIFTFeatures(img_gray);
        if points.Count >= 2
            [featSIFT, ~] = extractFeatures(img_gray, points);
            featSIFT = double(featSIFT);
            % Utilitzem la mitjana dels descriptors SIFT com a representació
            siftFeat = mean(featSIFT, 1);
            
            % Normalització L2 per evitar valors massa grans
            normVal = norm(siftFeat);
            if normVal > 0
                siftFeat = siftFeat / normVal;
            end
            
            % Agafem només les primeres 32 dimensions
            if numel(siftFeat) > siftDim
                siftFeat = siftFeat(1:siftDim);
            elseif numel(siftFeat) < siftDim
                siftFeat = [siftFeat, zeros(1, siftDim - numel(siftFeat))];
            end
        end
    catch
        % Si SIFT falla (falta toolbox, etc.), deixem zeros
        siftFeat = zeros(1, siftDim);
    end

    feat = [feat, siftFeat];
end