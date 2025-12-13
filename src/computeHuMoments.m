function hu = computeHuMoments(bw)
    hu = zeros(1, 7);
    [y, x] = find(bw);
    if isempty(x), return; end
    
    % Centroids i moments centrals
    xc = mean(x); yc = mean(y);
    
    % Helper intern per moment central
    mu_calc = @(p,q) sum((x-xc).^p .* (y-yc).^q);
    
    mu20 = mu_calc(2,0); mu02 = mu_calc(0,2); mu11 = mu_calc(1,1);
    mu30 = mu_calc(3,0); mu03 = mu_calc(0,3); 
    mu21 = mu_calc(2,1); mu12 = mu_calc(1,2);
    
    % Normalització
    n = length(x);
    eta = @(m, p, q) m / (n^((p+q)/2 + 1));
    
    n20 = eta(mu20,2,0); n02 = eta(mu02,0,2); n11 = eta(mu11,1,1);
    n30 = eta(mu30,3,0); n03 = eta(mu03,0,3); 
    n21 = eta(mu21,2,1); n12 = eta(mu12,1,2);
    
    % 7 Moments de Hu
    hu(1) = n20 + n02;
    hu(2) = (n20 - n02)^2 + 4*n11^2;
    hu(3) = (n30 - 3*n12)^2 + (3*n21 - n03)^2;
    hu(4) = (n30 + n12)^2 + (n21 + n03)^2;
    hu(5) = (n30 - 3*n12)*(n30 + n12)*((n30 + n12)^2 - 3*(n21 + n03)^2) + ...
            (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)^2 - (n21 + n03)^2);
    hu(6) = (n20 - n02)*((n30 + n12)^2 - (n21 + n03)^2) + ...
            4*n11*(n30 + n12)*(n21 + n03);
    hu(7) = (3*n21 - n03)*(n30 + n12)*((n30 + n12)^2 - 3*(n21 + n03)^2) - ...
            (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)^2 - (n21 + n03)^2);
            
    % Log transform
    hu = -sign(hu) .* log10(abs(hu) + 1e-10);
end