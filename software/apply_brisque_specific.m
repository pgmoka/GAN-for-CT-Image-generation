function returnMe = apply_brisque_specific(fname)
    fid = fopen(fname,'r');
    img = fread(fid, 'float');
    img = reshape(img, 512,512);   %or other dimensions if you are reading a sinogram)
    fclose(fid);
    imagesc(img);
 
%     ar = imread(fname);
    returnMe = brisque(img);
end