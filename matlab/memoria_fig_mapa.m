% Figura de la memoria: el mapa embarcado.
%
% Genera figs/fig_mapa.png para el capitulo 2, con los 258 crateres del mapa
% embarcado dibujados a su tamano real. mapa_referencia.csv no esta en la
% carpeta matlab sino en blender, asi que la ruta se resuelve desde la raiz
% del repositorio.

clc; clear; close all;

aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
csvMap = fullfile(raiz, 'blender', 'mapa_referencia.csv');
salida = fullfile(raiz, 'latex', 'figs', 'fig_mapa.png');

if ~exist(fileparts(salida), 'dir')
    mkdir(fileparts(salida));
end

M = readtable(csvMap);
fprintf('%d crateres, D de %.1f a %.1f km\n', height(M), min(M.D_km), max(M.D_km));

% Extension del terreno (la del catalogo completo, no la del mapa)
ANCHO_KM = 840;
ALTO_KM  = 630;

fig = figure('Color','w', 'Units','centimeters', 'Position',[2 2 18 14]);
ax  = axes(fig); hold(ax,'on'); axis(ax,'equal')

% Borde del terreno
rectangle('Position',[0 0 ANCHO_KM ALTO_KM], 'EdgeColor',[.75 .75 .78], ...
          'LineWidth',0.8);

% Un circulo por crater, a escala real
th = linspace(0, 2*pi, 72);
for i = 1:height(M)
    r = M.D_km(i)/2;
    fill(M.x_km(i) + r*cos(th), M.y_km(i) + r*sin(th), [.86 .86 .89], ...
         'EdgeColor',[.42 .42 .48], 'LineWidth',0.5);
end

xlabel('x (km)'); ylabel('y (km)')
xlim([-10 ANCHO_KM+10]); ylim([-10 ALTO_KM+10])
box on; grid on
set(ax, 'FontSize',9, 'GridAlpha',0.10, 'Layer','top')

% Escala grafica de 100 km, mas util que fiarse de los ejes
plot([40 140], [30 30], 'k-', 'LineWidth',2);
text(90, 48, '100 km', 'HorizontalAlignment','center', 'FontSize',9);

exportgraphics(fig, salida, 'Resolution', 300);
fprintf('Figura guardada en:\n  %s\n', salida);
