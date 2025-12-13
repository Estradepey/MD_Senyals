function models = trainModels(X, Y)
    % Comprovació de seguretat: Tenim dades?
    if isempty(X) || isempty(Y)
        error('Error crític: Les matrius X o Y estan buides. Revisa extractDatasetFeatures per veure si estàs filtrant totes les imatges (NaNs).');
    end

    % 1. Netejar etiquetes (eliminar categories sense dades)
    Y = removecats(Y); 
    
    % 2. Calcular Pesos (Class Weights)
    classes = categories(Y);
    numClasses = numel(classes);
    nTotal = length(Y);
    weights = ones(nTotal, 1);
    
    fprintf('  Calculant pesos per a %d classes...\n', numClasses);
    
    for i = 1:numClasses
        c = classes{i};
        idx = (Y == c);
        nCount = sum(idx);
        
        if nCount > 0
            % Fórmula: Pes inversament proporcional a la freqüència
            w = nTotal / (numClasses * nCount);
            weights(idx) = w;
        else
            warning('La classe "%s" no té mostres al Train set!', char(c));
        end
    end
    
    % Normalitzar pesos perquè sumin nTotal (opcional però recomanable per estabilitat)
    weights = weights / mean(weights);

    %% ENTRENAMENT DELS MODELS
    
    % 1. SVM (amb Pesos i ClassNames explícits)
    fprintf('  -> SVM (Weighted)...\n');
    t = templateSVM('KernelFunction', 'rbf', 'Standardize', true);
    
    % Passem 'ClassNames' explícitament per evitar l'error "No class names found"
    models.svm = fitcecoc(X, Y, ...
        'Learners', t, ...
        'Weights', weights, ...
        'ClassNames', classes); 
    
    % 2. Random Forest (amb Pesos)
    fprintf('  -> Random Forest (Weighted)...\n');
    models.rf = TreeBagger(100, X, Y, ...
        'Method', 'classification', ...
        'Weights', weights, ...
        'MinLeafSize', 3, ... % Fulles petites per classes petites
        'OOBPrediction', 'on');
    
    % 3. KNN
    fprintf('  -> KNN...\n');
    models.knn = fitcknn(X, Y, ...
        'NumNeighbors', 5, ... % Reduït a 5 per ser més sensible localment
        'Standardize', true, ...
        'Weights', weights); % KNN també accepta pesos!
    
    models.names = {'SVM Weighted', 'Random Forest Weighted', 'KNN Weighted'};
end