function [results, bestName, bestIdx] = evaluateModels(models, XTest, YTest)
    % Prediccions
    predSVM = predict(models.svm, XTest);
    predRF  = categorical(predict(models.rf, XTest));
    predKNN = predict(models.knn, XTest);
    
    % Càlcul Accuracy
    accSVM = sum(predSVM == YTest) / numel(YTest) * 100;
    accRF  = sum(predRF == YTest)  / numel(YTest) * 100;
    accKNN = sum(predKNN == YTest) / numel(YTest) * 100;
    
    fprintf('\nRESULTATS:\n');
    fprintf('SVM: %.2f%% | RF: %.2f%% | KNN: %.2f%%\n', accSVM, accRF, accKNN);
    
    % Empaquetar resultats
    results.preds = {predSVM, predRF, predKNN};
    results.accs = [accSVM, accRF, accKNN];
    
    [~, bestIdx] = max(results.accs);
    bestName = models.names{bestIdx};
    fprintf('MILLOR MODEL: %s\n', bestName);
end