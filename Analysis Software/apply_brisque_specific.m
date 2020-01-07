%  Made by Pedro Goncalves Mokarzel
%  while attending UW Bothell Student ID# 1576696
%  Made in 12/09/2019
%  Based on instruction in CSS 490, 
%  taught by professor Dong Si

% Precondition: path of float to apply brisque
% Postcondition: int with brisque score
function returnMe = apply_brisque_specific(fname)
    fid = fopen(fname,'r');
    img = fread(fid, 'float');
    img = reshape(img, 512,512);   %or other dimensions if you are reading a sinogram)
    fclose(fid);
%     imagesc(img);
 
%     ar = imread(fname);
    returnMe = brisque(img);
end