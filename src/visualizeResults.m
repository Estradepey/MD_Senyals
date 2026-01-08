function visualizeResults(results, bestIdx, models, imdsTest, YTest)
    % Mostra gràfiques de rendiment i exemples
    bestModelName = models.names{bestIdx};
    bestPreds = results.preds{bestIdx};
    bestAcc = results.accs(bestIdx);
    
    fprintf('Generant gràfics per al model: %s...\n', bestModelName);

    %% 1. Matriu de Confusió
    figure('Name', 'Matriu de Confusió', 'NumberTitle', 'off', 'Position', [100 100 800 600]);
    cm = confusionchart(YTest, bestPreds);
    cm.Title = sprintf('Matriu de Confusió - %s (%.2f%%)', bestModelName, bestAcc);
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';

    %% 2. Mètriques Detallades (Precision, Recall, F1)
    figure('Name', 'Mètriques per Classe', 'NumberTitle', 'off', 'Position', [100 100 1200 400]);
    classes = categories(YTest);
    nClasses = length(classes);
    precision = zeros(nClasses, 1);
    recall = zeros(nClasses, 1);
    f1score = zeros(nClasses, 1);

    for i = 1:nClasses
        cls = classes{i};
        % Càlcul manual de mètriques one-vs-rest
        tp = sum(bestPreds == cls & YTest == cls);
        fp = sum(bestPreds == cls & YTest ~= cls);
        fn = sum(bestPreds ~= cls & YTest == cls);
        
        precision(i) = tp / (tp + fp + eps);
        recall(i) = tp / (tp + fn + eps);
        f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
    end

    % Subplots
    subplot(1,3,1); bar(precision); title('Precision'); ylim([0 1]);
    xtickangle(45); xticklabels(classes);
    
    subplot(1,3,2); bar(recall); title('Recall'); ylim([0 1]);
    xtickangle(45); xticklabels(classes);
    
    subplot(1,3,3); bar(f1score); title('F1-Score'); ylim([0 1]);
    xtickangle(45); xticklabels(classes);
    
    sgtitle(sprintf('Mètriques per Classe - %s', bestModelName));

    %% 3. Exemples Visuals
    
    if numel(imdsTest.Files) == numel(YTest)
        figure('Name', 'Exemples de Predicció', 'NumberTitle', 'off', 'Position', [100 100 1200 800]);
        nExamples = min(12, numel(YTest));
        rng(42); % Llavor fixa per reproductibilitat
        idxs = randperm(numel(YTest), nExamples);
        
        for i = 1:nExamples
            idx = idxs(i);
            subplot(3, 4, i);
            
            % Llegir imatge original
            img = readimage(imdsTest, idx);
            imshow(imresize(img, [128 128]));
            
            actual = char(YTest(idx));
            predicted = char(bestPreds(idx));
            isCorrect = strcmp(actual, predicted);
            
            if isCorrect
                title(sprintf('✓ %s', actual), 'Color', [0 0.7 0], 'FontSize', 10, 'FontWeight', 'bold');
            else
                title(sprintf('✗ Real: %s\nPred: %s', actual, predicted), 'Color', [0.8 0 0], 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
        sgtitle(sprintf('Prediccions Aleatòries - %s', bestModelName));
    else
        warning('El nombre d''imatges i etiquetes no coincideix (possiblement per filtratge d''errors). Es salta la visualització d''imatges.');
    end
end