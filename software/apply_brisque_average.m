% Precondition: string with path to directory to be analyzed
% Postcondition: int with the average of the BRISQUE
% between all png in the file path
function returnMe = apply_brisque_average(path)
    files_names = dir([path '*.flt']);
    returnMe = 0;
    for k=1:length(files_names)
        name = [path files_names(k).name];
        fid = fopen(name,'r');
        ar = fread(fid, 'float');
        ar = reshape(ar, 512,512);   %or other dimensions if you are reading a sinogram)
        fclose(fid);
        % Adds to variable
        returnMe = returnMe + brisque(ar);
    end
    % Calculates average
    returnMe = returnMe / length(files_names);
end
