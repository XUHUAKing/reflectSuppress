fileList = dir('../figures/*.png');
for idx = 1:length(fileList)
    file = fileList(idx);
    savedir = fullfile('../results/', file.name);
    datadir = fullfile('../figures/', file.name);
    Im = imread(datadir);
    T = reflectSuppress(Im, 0.033, 1e-8);
    Gray = rgb2gray(T); 
    % save grayscale output in 16 bit after multiple 65,535
    Gray = uint16(Gray * 65535);
    Gray(Gray<0) = 0;
    Gray(Gray>65535) = 65535;
    imwrite(Gray, savedir);
end
subplot(1,3,1); 
imshow(Im); 
subplot(1,3,2); 
imshow(T);
subplot(1,3,3); 
imshow(Gray);
