function feat = extractSingleFeature(img, config)
    %% 0. PRE-PROCESSAMENT
    % Si la imatge ve en format 16 bits (0-65535), la passem a 8 bits (0-255)
    if isa(img, 'uint16')
        img = im2uint8(img);
    end
    %(MIDA FIXA)
    img = imresize(img, [config.imgSize config.imgSize]);
    
    img_gray = rgb2gray(img);
    
    %(Vores)
    bw = edge(img_gray, 'canny', [0.1 0.3]);
    bw = imclose(bw, strel('disk', 2));
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
end