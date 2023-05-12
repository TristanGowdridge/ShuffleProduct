function f = partial_fractions(filename)
syms x0

% If a file of this name already exists, delete it to not cause repeats in
% the data.

if exist(strcat(filename, '_MATLAB.txt'), 'file') == 2
    delete(strcat(filename, '_MATLAB.txt'));
end
fid = fopen(strcat(filename, '_python.txt'));

while ~feof(fid)
    term = eval(fgetl(fid));
    if isa(term, 'sym')
        term = partfrac(term);
    end
    term_str = string(term).replace('^', "**").replace('i', 'j');
    writelines(term_str, strcat(filename, "_MATLAB.txt"), WriteMode="append");
end
fclose(fid);

end