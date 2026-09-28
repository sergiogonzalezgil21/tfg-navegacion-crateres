% Fase 3A: trayectoria tridimensional sobre el terreno.
%
% Dibuja la trayectoria real y la estimada en 3D sobre la superficie generada,
% con su textura. Requiere haber ejecutado antes fase3a_offline.
%
% altura.png se guardo normalizado entre el minimo y el maximo del relieve,
% asi que por si solo no dice cuantos metros vale un nivel de gris. La escala
% se recupera calibrando contra el catalogo: crateres_catalogo.csv da la
% posicion y la z_km de 41.702 crateres, se muestrea el mapa de alturas en
% esas posiciones y se ajusta z_km = a*Hn + b por minimos cuadrados. El script
% imprime el R2 del ajuste.
%
% El relieve es de pocos km sobre un mapa de 840 km de ancho, asi que solo el
% terreno se exagera verticalmente, y queda indicado en el eje. La altitud de
% la nave va en kilometros verdaderos.

clc; close all;

%% ------------------------------------------------------------------ rutas
aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
rutaAlt  = fullfile(raiz, 'blender', 'altura.png');
rutaAlb  = fullfile(raiz, 'blender', 'albedo.png');
rutaCat  = fullfile(raiz, 'blender', 'crateres_catalogo.csv');
rutaReal = fullfile(raiz, 'blender', 'trayectoria_real.csv');
rutaRes  = fullfile(raiz, 'matlab',  'fase3a_resultados.csv');

ANCHO_KM = 840;
ALTO_KM  = 630;

PASO        = 6;      % submuestreo del terreno; 6 va fino y rapido
CON_TEXTURA = true;   % false = relieve en escala de grises, mas ligero
EXAG_RELIEVE = 8;     % exageracion SOLO del terreno (la altitud va real)

%% ------------------------------------------------------- mapa de alturas
fprintf('Leyendo el mapa de alturas...\n');
Hn = imread(rutaAlt);
if size(Hn,3) > 1, Hn = Hn(:,:,1); end
if isa(Hn,'uint16'), Hn = double(Hn)/65535; else, Hn = double(Hn)/255; end
[nf, nc] = size(Hn);
fprintf('  %d x %d pixeles\n', nc, nf);

%% ------------------------------------------- calibracion contra catalogo
T  = readtable(rutaCat);
col = round(T.x_km / ANCHO_KM * nc + 0.5);
fil = round((1 - T.y_km / ALTO_KM) * nf + 0.5);   % la fila 1 es y = ALTO
val = col >= 1 & col <= nc & fil >= 1 & fil <= nf;
v   = Hn(sub2ind([nf nc], fil(val), col(val)));
zk  = T.z_km(val);

M    = [v(:), ones(numel(v),1)];
coef = M \ zk(:);
a = coef(1);  b = coef(2);
res = zk(:) - M*coef;
R2  = 1 - sum(res.^2) / sum((zk(:)-mean(zk)).^2);

fprintf('Calibracion con %d crateres:\n', numel(v));
fprintf('  z_km = %.4f * Hn %+.4f      R2 = %.4f\n', a, b, R2);
fprintf('  relieve de %.2f a %.2f km  (rango %.2f km)\n', b, a+b, a);
if R2 < 0.5
    warning(['Ajuste flojo: puede que el catalogo guarde la cota previa ' ...
             'al crater y no la del terreno final.']);
end

%% ------------------------------------------------------------- superficie
% Se reduce PROMEDIANDO cada bloque de PASO x PASO, no cogiendo un pixel
% de cada PASO. Con el submuestreo simple aparecen puas: un crater de
% 10 km mide 8 px en este mapa, asi que si el pixel elegido cae en su
% fondo, ese vertice se hunde solo y sale un pincho que no existe.
%
% Se voltea ademas para que la fila 1 sea y = 0: interp2 necesita los
% ejes crecientes, y asi la textura queda alineada sin sorpresas.
Zs = (bloque_media(flipud(Hn), PASO) * a + b) * EXAG_RELIEVE;
[nf2, nc2] = size(Zs);
[Xs, Ys] = meshgrid(linspace(0, ANCHO_KM, nc2), linspace(0, ALTO_KM, nf2));

%% ------------------------------------------------------------ trayectoria
Tr = readtable(rutaReal);
Re = readtable(rutaRes);
ok = Re.valido == 1;

zSuelo = @(x,y) interp2(Xs, Ys, Zs, x, y, 'linear', 0);

%% ------------------------------------------------------------------ plot
figure('Color','w','Name','Trayectoria 3D','Position',[60 60 1250 720]);
hold on;

s = surf(Xs, Ys, Zs, 'EdgeColor', 'none', 'HandleVisibility', 'off');
if CON_TEXTURA && isfile(rutaAlb)
    Alb = imread(rutaAlb);
    if size(Alb,3) == 1, Alb = repmat(Alb, 1, 1, 3); end
    % OJO: nada de 'shading interp' aqui, sobreescribiria FaceColor y se
    % cargaria la textura.
    set(s, 'CData', flipud(Alb), 'FaceColor', 'texturemap');
else
    colormap(gray(256));
    set(s, 'FaceColor', 'interp', 'CData', Zs);
end

% traza sobre el suelo, para leer la posicion horizontal
h1 = plot3(Tr.x_km, Tr.y_km, zSuelo(Tr.x_km, Tr.y_km) + 3, ...
    '-', 'Color', [1 1 1], 'LineWidth', 1.0);

% verticales, para ver la altura de un vistazo
idx = round(linspace(1, height(Tr), 14));
for i = idx
    plot3([Tr.x_km(i) Tr.x_km(i)], [Tr.y_km(i) Tr.y_km(i)], ...
          [zSuelo(Tr.x_km(i), Tr.y_km(i)), Tr.altura_km(i)], ...
          '-', 'Color', [.65 .65 .65], 'LineWidth', 0.5, ...
          'HandleVisibility', 'off');
end

h2 = plot3(Tr.x_km, Tr.y_km, Tr.altura_km, ...
    '-', 'Color', [.35 .60 .88], 'LineWidth', 3.0);
% linea CONTINUA, no discontinua: con rayas, los huecos dejan ver el azul
% de debajo y la estimada parece rota cuando en realidad se solapa.
h3 = plot3(Re.x_km(ok), Re.y_km(ok), Re.altura_km(ok), ...
    '-', 'Color', [.85 .20 .28], 'LineWidth', 1.1);

h4 = plot3(Tr.x_km(1), Tr.y_km(1), Tr.altura_km(1), 'o', ...
    'MarkerSize', 9, 'MarkerFaceColor', [.16 .43 .73], 'MarkerEdgeColor', 'w');
plot3(Tr.x_km(end), Tr.y_km(end), Tr.altura_km(end), 's', ...
    'MarkerSize', 9, 'MarkerFaceColor', [.16 .43 .73], ...
    'MarkerEdgeColor', 'w', 'HandleVisibility', 'off');

grid on; box on;
xlabel('x (km)');
ylabel('y (km)');
zlabel(sprintf('z (km)   -   relieve exagerado x%d', EXAG_RELIEVE));

% ---------------------------------------------------------------------
% OJO CON EL ASPECTO. Aqui hay tres escalas muy distintas conviviendo:
% el mapa mide 840 x 630 km, el relieve son +-5 km y la nave vuela a
% 120-170 km. Si se usa daspect (relacion de los DATOS) con los tres
% limites fijados a mano, el problema queda sobredeterminado: MATLAB da
% prioridad al aspecto y REAJUSTA LOS LIMITES, encogiendo x e y para que
% cuadren con el rango pequeno de z. El resultado es un parche central
% del terreno y la trayectoria fuera del recuadro.
%
% Se usa pbaspect, que fija la forma del RECUADRO y no toca los datos, y
% los limites se ponen DESPUES para que no los pise nadie.
% ---------------------------------------------------------------------
pbaspect([1.33 1 0.55]);
view(-38, 26);
camlight(-40, 55); lighting gouraud; material dull;

xlim([0 ANCHO_KM]);
ylim([0 ALTO_KM]);
zlim([min(Zs(:)) - 5, max(Tr.altura_km) * 1.10]);

legend([h2 h3 h1 h4], ...
    {'trayectoria real','estimada','traza en el suelo','inicio / fin'}, ...
    'Location', 'northwest');
title(sprintf('Trayectoria 3D  |  %d/%d frames  |  error mediano %.3f km', ...
    sum(ok), height(Re), median(Re.error_pos_km(ok), 'omitnan')));

fprintf('\nHecho. Rota la figura con la herramienta 3D de la barra.\n');
fprintf('Vista rasante:      view(-62, 8)\n');
fprintf('Recuadro mas plano: pbaspect([1.33 1 0.30])\n');
fprintf('Vista cenital:      view(0, 90)\n');
fprintf('Aplastar el relieve: cambia EXAG_RELIEVE arriba\n');


% =========================================================================
function B = bloque_media(A, k)
% Reduce A promediando bloques de k x k. Sin toolboxes.
% Frente al submuestreo A(1:k:end,1:k:end), evita las puas que salen
% cuando el pixel elegido cae en el fondo de un crater pequeno.
if k <= 1, B = A; return; end
[nf, nc] = size(A);
nf2 = floor(nf/k)*k;
nc2 = floor(nc/k)*k;
A = A(1:nf2, 1:nc2);
B = reshape(A, k, nf2/k, k, nc2/k);
B = squeeze(mean(mean(B, 1), 3));
end
