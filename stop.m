% 1. Apuntar a la carpeta principal on tens les subcarpetes
datasetPath = 'C:\Users\max.estrade\Downloads\imatges_senyals'; 

% 2. Crear el datastore. 
% 'LabelSource', 'foldernames' fa que l'etiqueta sigui el nom de la carpeta.
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 3. Separar en Train (70%) i Test (30%) de forma aleatòria
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

% Comprovació
countEachLabel(imdsTrain) % Et mostra quantes imatges tens per entrenar

numImages = numel(imdsTrain.Files);
numFeatures = 20; % Exemple: farem servir els primers 20 descriptors de Fourier

% Pre-assignar espai per guanyar velocitat
XTrain = zeros(numImages, numFeatures); 
YTrain = imdsTrain.Labels; % Les etiquetes ja les tenim

% Bucle per processar cada imatge
for i = 1:numImages
    % A. Llegir la imatge
    img = readimage(imdsTrain, i);
    
    % --- INICI DE LA TEVA PART DE PROCESSAMENT ---
    
    % B. Preprocessament (Binarització)
    % Adapta això segons el que funcioni millor (per color o vores)
    img_gray = rgb2gray(img);
    bw = edge(img_gray, 'canny'); 
    % (Aquí hauries d'afegir neteja morfològica imclose/imfill)
    
    % C. Extracció de Característiques (Exemple: Fourier de E15.pdf)
    % 1. Trobar contorn
    [contorn_y, contorn_x] = find(bw, 1);
    if isempty(contorn_y) 
        % Si no troba res, posem zeros (o gestionem l'error)
        feat = zeros(1, numFeatures);
    else
        boundary = bwtraceboundary(bw, [contorn_y, contorn_x], 'N');
        
        % 2. Complexificar i FFT
        s = boundary(:,2) + 1i * boundary(:,1);
        z = fft(s);
        
        % 3. Normalitzar (Invariant a escala i rotació)
        % (Important descartar el primer coeficient z(1) que depèn de la posició)
        z_magnitud = abs(z);
        z_norm = z_magnitud(2:end); % Treiem el DC component
        if ~isempty(z_norm)
             z_norm = z_norm / z_norm(1); % Normalitzar respecte al primer harmònic
        end
        
        % 4. Agafar els primers N descriptors
        if length(z_norm) >= numFeatures
            feat = z_norm(1:numFeatures).';
        else
            % Padding si el contorn és molt petit
            feat = [z_norm.', zeros(1, numFeatures - length(z_norm))];
        end
    end
    
    % --- FINAL DEL PROCESSAMENT ---
    
    % D. Guardar a la matriu
    XTrain(i, :) = feat;
end

disp('Extracció de característiques completada.');

T = array2table(XTrain);
T.Label = YTrain;