function [patches, bboxes] = detectAndSegmentSigns(fullImage, config)
    % detectAndSegmentSigns: Escanea una imagen completa buscando señales.
    %
    % INPUT:
    %   - fullImage: Imagen original (RGB)
    %   - config: Estructura de configuración (opcional, por si quieres pasar parámetros)
    %
    % OUTPUT:
    %   - patches: Cell array con las imágenes recortadas candidatas
    %   - bboxes: Matriz Nx4 con las coordenadas [x, y, w, h] de cada parche
    
    patches = {};
    bboxes = [];
    
    % 1. VALIDACIÓN
    if size(fullImage, 3) ~= 3
        warning('La imagen no es RGB. No se puede segmentar por color.');
        return;
    end
    
    % 2. PRE-PROCESADO Y HSV
    % Convertimos a HSV para ser inmunes a sombras/brillo
    img_hsv = rgb2hsv(fullImage);
    
    % --- MÁSCARAS DE COLOR ---
    
    % ROJO: El rojo está en los dos extremos del Hue (0 y 1)
    % Rango 1: 0.90 a 1.0
    mask_r1 = (img_hsv(:,:,1) >= 0.90) & (img_hsv(:,:,1) <= 1.0);
    % Rango 2: 0.00 a 0.10
    mask_r2 = (img_hsv(:,:,1) >= 0.00) & (img_hsv(:,:,1) <= 0.10);
    % Saturación alta (> 0.4) para evitar grises/blancos
    mask_sat = (img_hsv(:,:,2) > 0.4);
    mask_red = (mask_r1 | mask_r2) & mask_sat;
    
    % AZUL: Señales de obligación/informativas (Hue aprox 0.55 - 0.75)
    mask_blue = (img_hsv(:,:,1) >= 0.55) & (img_hsv(:,:,1) <= 0.75) & (img_hsv(:,:,2) > 0.4);
    
    % Unir máscaras
    binaryMask = mask_red | mask_blue;
    
    % 3. LIMPIEZA MORFOLÓGICA (Morphology)
    % Eliminar ruido (píxeles sueltos)
    binaryMask = bwareaopen(binaryMask, 30); 
    % Dilatar para unir trozos rotos de la señal
    binaryMask = imdilate(binaryMask, strel('disk', 2));
    % Rellenar agujeros (el centro de la señal)
    binaryMask = imfill(binaryMask, 'holes');
    
    % 4. EXTRACCIÓN DE PROPIEDADES
    stats = regionprops(binaryMask, 'BoundingBox', 'Area', 'Eccentricity', 'Extent', 'Solidity');
    
    % 5. FILTRADO DE CANDIDATOS
    imgArea = size(fullImage, 1) * size(fullImage, 2);
    
    for k = 1:length(stats)
        bb = stats(k).BoundingBox; % [x, y, w, h]
        w = bb(3);
        h = bb(4);
        area = stats(k).Area;
        aspectRatio = w / h;
        
        % --- REGLAS DE RECHAZO ---
        
        % A. Tamaño: Ni muy pequeño (ruido) ni gigante (cielo/suelo)
        if area < 200 || area > (imgArea / 3)
            continue; 
        end
        
        % B. Forma: Las señales suelen ser cuadradas (1:1). 
        % Aceptamos de 0.6 a 1.5 por perspectiva.
        if aspectRatio < 0.6 || aspectRatio > 1.6
            continue;
        end
        
        % C. Solidez (Solidity): Proporción de píxeles dentro de la forma convexa
        % Un cuadrado o círculo es muy sólido (>0.9). Una mancha irregular es baja.
        if stats(k).Solidity < 0.85
            continue; % Descartar formas raras
        end
    
        % D. Extent (Opcional): Cuánto de la caja ocupa el objeto
        % Un círculo dentro de un cuadrado ocupa pi/4 = 0.78. 
        % Si ocupa menos de 0.5, es que hay mucho aire (posiblemente ruido).
        if stats(k).Extent < 0.5
            continue;
        end
        
        % SI PASA LOS FILTROS -> ES UN CANDIDATO
        
        % Recortamos (Patching)
        % Añadimos un pequeño margen (padding) si es posible para no cortar el borde
        padding = 2;
        rect = [bb(1)-padding, bb(2)-padding, bb(3)+2*padding, bb(4)+2*padding];
        patch = imcrop(fullImage, rect);
        
        if ~isempty(patch)
            patches{end+1} = patch; %#ok<AGROW>
            bboxes = [bboxes; bb];   %#ok<AGROW>
        end
    end
end