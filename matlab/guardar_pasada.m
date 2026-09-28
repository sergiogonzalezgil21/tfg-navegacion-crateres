function guardar_pasada(n)
% GUARDAR_PASADA  Archiva los CSV de resultados de una ejecucion en
% matlab\pasadas\pasada_<n>\.
%
%   guardar_pasada(1)
%
% Las fases escriben siempre sobre el mismo nombre de archivo, asi que cada
% ejecucion borra la anterior. Conservarlas permite medir la repetibilidad del
% sistema: la inferencia de YOLO en GPU no es determinista y el numero de
% fotogramas resueltos cambia entre ejecuciones identicas.

if nargin < 1 || ~isscalar(n) || n < 1
    error('Uso: guardar_pasada(n) con n entero >= 1');
end

aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
dirML = fullfile(raiz, 'matlab');
destino = fullfile(dirML, 'pasadas', sprintf('pasada_%d', n));
if ~isfolder(destino), mkdir(destino); end

archivos = {'fase3a_resultados.csv', ...
            'fase3b_resultados.csv', ...
            'fase4_directo_resultados.csv', ...
            'fase4_cascada.csv'};

fprintf('Archivando en %s\n', destino);
for i = 1:numel(archivos)
    origen = fullfile(dirML, archivos{i});
    if ~isfile(origen)
        fprintf('  (falta)   %s\n', archivos{i});
        continue;
    end
    info = dir(origen);
    T = readtable(origen);
    copyfile(origen, fullfile(destino, archivos{i}));
    fprintf('  guardado  %-32s %4d filas   %s\n', archivos{i}, height(T), ...
            datestr(info.datenum, 'dd/mm/yyyy HH:MM'));
end
end
