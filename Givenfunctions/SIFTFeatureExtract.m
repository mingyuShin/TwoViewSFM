function [finalKp1, finalKp2] = SIFTFeatureExtract(img1, img2)
    grayI1 = rgb2gray(img1);
    grayI2 = rgb2gray(img2);
    
    kp1 = detectSIFTFeatures(grayI1);
    kp2 = detectSIFTFeatures(grayI2);

    [des1, pts1] = extractFeatures(grayI1, kp1); %(2581,128), (2581,1)
    [des2, pts2] = extractFeatures(grayI2, kp2);
    
    matches = zeros(size(des1, 1), 2);
    matches(:,1) = 1:size(des1, 1);

    [idx, d] = knnsearch(des2, des1, 'k', 2, 'NSMethod', 'kdtree');
    matches(:,2) = idx(:,1);
    
    confidences = d(:,1) ./ d(:,2);

    [confidences, ind] = sort(confidences, 'ascend');
    cidx = find((confidences<0.6)==0, 1);
    matches = matches(ind, :);
    matches = matches(1:cidx, :);
    matchedI1 = pts1(matches(:,1));
    matchedI2 = pts2(matches(:,2));
    
    finalKp1 = matchedI1.Location;
    finalKp2 = matchedI2.Location;
    
%     figure(2)
%     showMatchedFeatures(img1, img2, matchedI1, matchedI2)
end