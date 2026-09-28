% Fase 4: monta el video del descenso a partir de los renders.
%
% Junta los 700 PNG de blender\Output_descenso en un unico .mp4 que consume
% fase4_directo.m.
%
% El detector se valido sobre los PNG originales, sin perdida. Con calidad 100
% la diferencia es despreciable, pero para reproducir exactamente los numeros
% de la memoria se puede poner PERFIL = 'Archival' (Motion JPEG 2000 sin
% perdida), unas 10 veces mas pesado.

clc; clear;

aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
dirPng  = fullfile(raiz, 'blender', 'Output_descenso');
salida  = fullfile(raiz, 'blender', 'vuelo_descenso.mp4');

FPS    = 25;            % ritmo nominal del fichero, no el de la demo
PERFIL = 'MPEG-4';      % o 'Archival' para sin perdida (cambia la extension a .mj2)
CALIDAD = 100;

L = dir(fullfile(dirPng, '*.png'));
if isempty(L)
    error('No hay PNG en %s', dirPng);
end
[~, ord] = sort({L.name});
L = L(ord);
fprintf('%d frames en %s\n', numel(L), dirPng);

v = VideoWriter(salida, PERFIL);
v.FrameRate = FPS;
if strcmp(PERFIL, 'MPEG-4')
    v.Quality = CALIDAD;
end
open(v);

tic;
for i = 1:numel(L)
    I = imread(fullfile(dirPng, L(i).name));
    if size(I,3) == 1, I = repmat(I, 1, 1, 3); end
    writeVideo(v, I);
    if mod(i, 100) == 0
        fprintf('  %d/%d  (%.2f s/frame)\n', i, numel(L), toc/i);
    end
end
close(v);

d = dir(salida);
fprintf('\nVideo listo: %s\n', salida);
fprintf('  %d frames, %.1f MB, %.1f s de duracion nominal\n', ...
    numel(L), d.bytes/1e6, numel(L)/FPS);
