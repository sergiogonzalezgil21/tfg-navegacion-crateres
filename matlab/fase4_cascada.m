% Fase 4: hasta que altura aguanta el emparejamiento por crateres.
%
% Navega el mismo vuelo de descenso con tres mapas de distinta densidad y mide
% a que altura se cae cada uno.
%
% El lado de la huella en el suelo es proporcional a la altura, asi que su
% area, y con ella el numero de crateres del mapa dentro del encuadre, cae con
% h^2. Cuando bajan de tres no hay tripleta que formar y el sistema se queda
% ciego. Ese limite no es del algoritmo sino de la densidad de landmarks del
% mapa embarcado, y enriquecer el mapa lo empuja hacia abajo: con D >= 10 km
% (258 crateres) la teoria situa el desplome sobre los 103 km, con D >= 6 km
% (440) sobre los 79 km y con D >= 4 km (753) sobre los 60 km. Es tambien la
% razon por la que los aterrizadores reales pasan a seguimiento de rasgos en
% el descenso final.
%
% Necesita, por este orden: render de CamaraDescenso en blender\Output_descenso,
% blender_exportar_descenso.py y 11_detecciones_descenso.py.

clc; clear; close all;

%% ------------------------------------------------------------------ rutas
aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
rutaDet  = fullfile(raiz, 'ia',      'detecciones_descenso.csv');
rutaReal = fullfile(raiz, 'blender', 'trayectoria_descenso.csv');
rutaOut  = fullfile(raiz, 'matlab',  'fase4_cascada.csv');

MAPAS = { 'mapa_referencia_D10.csv', 10, [.20 .55 .90]
          'mapa_referencia_D6.csv',   6, [.95 .65 .15]
          'mapa_referencia_D4.csv',   4, [.85 .25 .30] };

%% ------------------------------------------------------------- parametros
params.frameWidth  = 1920;
params.frameHeight = 1080;
params.FOV_horizontal = 60;
params.mapa_ancho_km = 840;
params.mapa_alto_km  = 630;

params.K = 7;
params.tol_forma = 0.02;
params.tol_diam  = 0.15;
% Rango de altitud plausible. NO es un detalle menor: junto con la
% comprobacion de tamano es lo unico que fija la ESCALA absoluta, porque el
% descriptor por si solo es invariante a escala. Con 20-400 km se colaban
% soluciones a 300 km con la escala tres veces mal. El vuelo va de 170 a 55,
% asi que 40-200 es holgado y a la vez realista: un aterrizador conoce su
% regimen de altitud por perfil de mision o por altimetro.
params.altura_min_km = 40;
params.altura_max_km = 200;
params.verif_px   = 55;
params.verif_diam = 0.30;
params.corrob_min = 4;

% El detector de la fase 4 (finetune_rotacion) devuelve cajas infladas
% segun el giro de la imagen, entre 1.16 y 1.45 veces el diametro real.
% Con esto la verificacion compara tamanos RELATIVOS en vez de absolutos.
params.corregir_inflado = true;
params.inflado_min = 0.85;    % el factor medido va de 1.10 a 1.45; fuera de
params.inflado_max = 1.70;    % este rango la hipotesis es imposible
params.cluster_km = 5;
params.salto_max_km = 45;
params.margen_hueco = 12;
params.radio_max_km = 400;

% poda acoplada a la escala: imprescindible con los mapas densos
PM.factor_lado = 30;
PM.lado_max_km = 225;

%% ------------------------------------------------------------------ datos
for p = {rutaDet, rutaReal}
    if ~isfile(p{1}), error('Falta %s', p{1}); end
end

Tr = readtable(rutaReal);
D  = leer_detecciones(rutaDet, params);
nF = height(Tr);

%% ------------------------------------------------------------ los 3 mapas
R = struct('nombre',{},'dmin',{},'color',{},'valido',{},'err',{}, ...
           'alt',{},'yaw',{},'n_mapa',{},'n_trip',{});

for im = 1:size(MAPAS,1)
    nombre = MAPAS{im,1};
    fprintf('\n===== %s =====\n', nombre);

    PMi = PM;
    PMi.cache = fullfile(raiz, 'matlab', ...
        sprintf('trip_D%d_f%d.mat', MAPAS{im,2}, PM.factor_lado));
    mapa = cargar_mapa(fullfile(raiz, 'blender', nombre), PMi);

    val = false(nF,1); err = nan(nF,1); alt = nan(nF,1); yaw = nan(nF,1);
    prev = []; last = []; hueco = 0;
    tic;

    for k = 1:nF
        j = k - D.frame_min + 1;
        if j < 1 || j > numel(D.datos) || isempty(D.datos{j})
            hueco = hueco + 1; continue;
        end

        if isempty(last)
            pred = []; radio = inf;
        else
            if isempty(prev), pred = last; else, pred = last + (last-prev); end
            radio = min(params.salto_max_km + hueco*params.margen_hueco, ...
                        params.radio_max_km);
        end

        S = resolver_frame(D.datos{j}, mapa, params, pred, radio);
        if isempty(S) && ~isempty(pred)
            S = resolver_frame(D.datos{j}, mapa, params, [], inf);
        end
        if isempty(S), hueco = hueco + 1; continue; end

        pnt = complex(S.x_km, S.y_km);
        if ~isempty(last) && abs(pnt - last) > radio
            hueco = hueco + 1; continue;
        end

        val(k) = true;
        err(k) = hypot(S.x_km - Tr.x_km(k), S.y_km - Tr.y_km(k));
        alt(k) = S.altura_km;
        yaw(k) = S.guinada_deg;
        prev = last; last = pnt; hueco = 0;

        if mod(k,150)==0, fprintf('  %d/%d\n', k, nF); end
    end

    fprintf('  resueltos %d/%d (%.1f %%) en %.0f s\n', ...
        sum(val), nF, 100*mean(val), toc);

    R(im).nombre = nombre;  R(im).dmin = MAPAS{im,2};
    R(im).color  = MAPAS{im,3};
    R(im).valido = val;     R(im).err = err;
    R(im).alt    = alt;     R(im).yaw = yaw;
    R(im).n_mapa = mapa.n;  R(im).n_trip = size(mapa.trip.ids,1);
end

%% ---------------------------------------------------- resueltos vs altura
bordes = 50:5:175;
centros = bordes(1:end-1) + 2.5;
hReal = Tr.altura_km;

fprintf('\n');
fprintf('%-8s', 'altura');
for im = 1:numel(R), fprintf('  D>=%-6d', R(im).dmin); end
fprintf('\n');

% OJO: hay que distinguir RESUELTO de CORRECTO.
%
% Por debajo del limite de landmarks el sistema no deja de dar respuesta:
% se engancha a una solucion equivocada y la da con toda la confianza. Se
% midio que el mapa de D>=4 km marcaba 100 % de frames "resueltos" entre 85
% y 100 km de altura mientras cometia errores de 300-400 km.
%
% La curva que significa algo es la de frames CORRECTOS. La de resueltos se
% dibuja tambien, en discontinuo: la distancia entre las dos es exactamente
% la tasa de mentiras del sistema, y esa brecha es en si misma un resultado.
TOL_OK = 5;    % km

PCT   = nan(numel(centros), numel(R));   % resueltos
PCTOK = nan(numel(centros), numel(R));   % correctos
for b = 1:numel(centros)
    m = hReal >= bordes(b) & hReal < bordes(b+1);
    if ~any(m), continue; end
    fprintf('%5.0f km', centros(b));
    for im = 1:numel(R)
        PCT(b,im)   = 100*mean(R(im).valido(m));
        PCTOK(b,im) = 100*mean(R(im).valido(m) & R(im).err(m) < TOL_OK);
        fprintf('  %5.0f/%3.0f%%', PCTOK(b,im), PCT(b,im));
    end
    fprintf('\n');
end
fprintf('  (correctos / resueltos, en %%.  correcto = error < %d km)\n', TOL_OK);

% altura a la que cada mapa baja del 50 %
fprintf('\nAltura de desplome (correctos por debajo del 50 %%):\n');
for im = 1:numel(R)
    v = PCTOK(:,im); c = centros(:);
    ok = ~isnan(v);
    idx = find(ok & v < 50, 1, 'last');
    if isempty(idx)
        fprintf('  D>=%2d km (%3d crateres, %7d tripletas): no se cae\n', ...
            R(im).dmin, R(im).n_mapa, R(im).n_trip);
    else
        fprintf('  D>=%2d km (%3d crateres, %7d tripletas): %.0f km\n', ...
            R(im).dmin, R(im).n_mapa, R(im).n_trip, c(idx));
    end
end

%% --------------------------------------------------------------- figuras
figure('Color','w','Name','Fase 4 - cascada','Position',[60 60 1500 800]);

subplot(2,2,1); hold on; grid on;
for im = 1:numel(R)     % discontinuo: lo que el sistema CREE que resuelve
    plot(centros, PCT(:,im), ':', 'Color', R(im).color, ...
        'LineWidth', 1.0, 'HandleVisibility', 'off');
end
for im = 1:numel(R)     % continuo: lo que resuelve DE VERDAD
    plot(centros, PCTOK(:,im), '-o', 'Color', R(im).color, ...
        'MarkerFaceColor', R(im).color, 'LineWidth', 1.8, 'MarkerSize', 4);
end
yline(50,'--','50 %','Color',[.4 .4 .4]);
set(gca,'XDir','reverse');      % el vuelo va de mas alto a mas bajo
xlabel('altitud verdadera (km)'); ylabel('frames resueltos (%)');
title(sprintf(['Hasta que altura aguanta cada mapa\n' ...
    'continuo = correctos (<%d km)   punteado = "resueltos"'], TOL_OK));
ylim([-2 102]);
legend(arrayfun(@(r) sprintf('D>=%d km (%d crateres)', r.dmin, r.n_mapa), ...
    R, 'UniformOutput', false), 'Location','southwest');

subplot(2,2,2); hold on; grid on;
for im = 1:numel(R)
    v = R(im).valido;
    plot(hReal(v), R(im).err(v), '.', 'Color', R(im).color, 'MarkerSize', 5);
end
set(gca,'XDir','reverse','YScale','log');
xlabel('altitud verdadera (km)'); ylabel('error de posicion (km)');
title('Error contra altitud');

subplot(2,2,3); hold on; grid on;
plot(Tr.frame, Tr.rot_z_deg, '-', 'Color', [.35 .60 .88], 'LineWidth', 2);
[~, mejor] = max(arrayfun(@(r) sum(r.valido), R));
v = R(mejor).valido;
plot(find(v), R(mejor).yaw(v), '.', 'Color', [.85 .25 .30], 'MarkerSize', 5);
xlabel('frame'); ylabel('guinada (deg)');
title(sprintf('Actitud: verdad contra estimada (D>=%d km)', R(mejor).dmin));
legend({'real','estimada'}, 'Location','best');

subplot(2,2,4); hold on; grid on;
plot(Tr.frame, Tr.altura_km, '-', 'Color', [.35 .60 .88], 'LineWidth', 2);
for im = 1:numel(R)
    v = R(im).valido;
    plot(find(v), R(im).alt(v), '.', 'Color', R(im).color, 'MarkerSize', 4);
end
xlabel('frame'); ylabel('altitud (km)');
title('Altitud: verdad contra estimada');

%% ---------------------------------------------- signo de la guinada y CSV
v = R(mejor).valido;
e1 = R(mejor).yaw(v) - Tr.rot_z_deg(v);
e2 = R(mejor).yaw(v) + Tr.rot_z_deg(v);
env = @(e) sqrt(mean((mod(e+180,360)-180).^2));
if env(e1) <= env(e2)
    fprintf('\nGuinada: est = +rot_z   RMS %.3f deg\n', env(e1));
else
    fprintf('\nGuinada: est = -rot_z   RMS %.3f deg  (signo invertido)\n', env(e2));
end

Tout = table(Tr.frame, Tr.altura_km, Tr.rot_z_deg, ...
    'VariableNames', {'frame','altura_real_km','guinada_real_deg'});
for im = 1:numel(R)
    s = sprintf('D%d', R(im).dmin);
    Tout.(['valido_' s]) = R(im).valido;
    Tout.(['error_' s])  = R(im).err;
    Tout.(['alt_' s])    = R(im).alt;
    Tout.(['yaw_' s])    = R(im).yaw;
end
writetable(Tout, rutaOut);
fprintf('Resultados en %s\n', rutaOut);
