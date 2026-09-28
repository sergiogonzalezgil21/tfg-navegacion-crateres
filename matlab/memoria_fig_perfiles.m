% Figura de la memoria: perfiles de las dos trayectorias.
%
% Genera figs/fig_perfiles.png para el capitulo 2: altitud y guinada de las
% dos camaras, fotograma a fotograma.

clc; clear; close all;

aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
dirBl  = fullfile(raiz, 'blender');
salida = fullfile(raiz, 'latex', 'figs', 'fig_perfiles.png');

% Por si la carpeta de figuras aun no existe
if ~exist(fileparts(salida), 'dir')
    mkdir(fileparts(salida));
end

C = readtable(fullfile(dirBl, 'trayectoria_real.csv'));       % crucero
D = readtable(fullfile(dirBl, 'trayectoria_descenso.csv'));   % descenso

% Colores validados para daltonismo y para impresion en gris. Ademas el
% crucero va con linea continua y el descenso discontinua, de modo que la
% figura se sigue leyendo aunque se imprima en blanco y negro.
colC = [0.122 0.435 0.749];   % azul - crucero  (#1F6FBF)
colD = [0.753 0.224 0.169];   % rojo - descenso (#C0392B)

fig = figure('Color','w', 'Units','centimeters', 'Position',[2 2 17 11]);

% ---------------- altitud ----------------
ax1 = subplot(2,1,1);
plot(C.frame, C.altura_km, '-',  'Color',colC, 'LineWidth',1.8); hold on
plot(D.frame, D.altura_km, '--', 'Color',colD, 'LineWidth',1.8);
grid on; box on
ylabel('altitud (km)')
xlim([1 max(C.frame)])
ylim([40 190])
legend({'crucero','descenso'}, 'Location','northeast', 'Box','off', ...
       'Orientation','horizontal')
title('Altitud')

% Nota: aqui NO se marca el limite de 118 km. Ese limite es un RESULTADO del
% capitulo 3 y esta figura pertenece al capitulo 2, que describe el metodo.

% ---------------- guiñada ----------------
ax2 = subplot(2,1,2);
plot(C.frame, C.rot_z_deg, '-',  'Color',colC, 'LineWidth',1.8); hold on
plot(D.frame, D.rot_z_deg, '--', 'Color',colD, 'LineWidth',1.8);
grid on; box on
xlabel('fotograma'); ylabel('guiñada (deg)')
xlim([1 max(C.frame)])
ylim([-200 200]); yticks(-180:90:180)
title('Guiñada')

set([ax1 ax2], 'FontSize',9, 'GridAlpha',0.12);
linkaxes([ax1 ax2], 'x');

% -dpng a 300 ppp da un PNG nitido en la memoria impresa
exportgraphics(fig, salida, 'Resolution', 300);

fprintf('Figura guardada en:\n  %s\n', salida);
fprintf('  crucero : altitud %.0f -> %.0f km, guiñada constante\n', ...
        C.altura_km(1), C.altura_km(end));
fprintf('  descenso: altitud %.0f -> %.0f km, guiñada %.0f a %.0f deg\n', ...
        D.altura_km(1), D.altura_km(end), min(D.rot_z_deg), max(D.rot_z_deg));
