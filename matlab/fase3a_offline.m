% Fase 3A: navegacion sobre frames ya renderizados.
% TFG - Navegacion optica relativa al terreno mediante lectura de crateres.
%
% Lee las detecciones que YOLO dejo en detecciones.csv, las empareja contra el
% mapa embarcado y reconstruye posicion, altitud y guinada frame a frame. Todo
% offline, con los frames en disco.
%
% El mapa y las salidas van en kilometros y los frames son de 1920x1080. El
% descriptor incluye los diametros ademas de la forma del triangulo, y la
% guinada se estima dentro del ajuste de similitud.
%
% Se corrigen dos sesgos sistematicos de altitud, medidos sobre el vuelo:
% cajas cortadas por el borde del frame (+2.85 km) y profundidad del centro de
% los crateres (+2.70 km).

clc; clear; close all;

%% ------------------------------------------------------------------ rutas
aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
rutaMapa = fullfile(raiz, 'blender', 'mapa_referencia.csv');
rutaDet  = fullfile(raiz, 'ia',      'detecciones.csv');
rutaReal = fullfile(raiz, 'blender', 'trayectoria_real.csv');   % solo validar
rutaOut  = fullfile(raiz, 'matlab',  'fase3a_resultados.csv');
rutaCache= fullfile(raiz, 'matlab',  'tripletas_mapa.mat');

%% ------------------------------------------------------------- parametros
params.frameWidth  = 1920;          % ANTES 640: los renders son 1920x1080
params.frameHeight = 1080;          % ANTES 480
params.FOV_horizontal = 60;

params.mapa_ancho_km = 840;
params.mapa_alto_km  = 630;

params.K = 7;                       % detecciones mas grandes que se usan
params.lado_max_km = 200;           % poda de tripletas que no caben en un frame

params.tol_forma = 0.02;            % tolerancia de los cocientes de lados
params.tol_diam  = 0.15;            % tolerancia de los cocientes de diametros

params.altura_min_km = 50;          % rango plausible, conocido a priori
params.altura_max_km = 400;

params.verif_px   = 55;             % tolerancia de verificacion, EN PIXELES
params.verif_diam = 0.30;           % el tamano tambien tiene que encajar
params.corrob_min = 4;              % crateres corroborados para aceptar
params.cluster_km = 5;

params.salto_max_km   = 45;         % entre frames consecutivos
params.margen_hueco   = 12;         % km que crece el radio por frame perdido
params.radio_max_km   = 400;

frameIni = 1;
frameFin = 700;

validar = isfile(rutaReal);

%% ------------------------------------------------------------------ carga
pm.lado_max_km = params.lado_max_km;
pm.cache = rutaCache;
mapa = cargar_mapa(rutaMapa, pm);

D = leer_detecciones(rutaDet, params);

if validar
    Treal = readtable(rutaReal);
    real_ = containers.Map(num2cell(Treal.frame), ...
        num2cell([Treal.x_km, Treal.y_km, Treal.altura_km], 2));
end

%% ------------------------------------------------------- bucle de frames
n = frameFin - frameIni + 1;
Res = table(zeros(n,1), false(n,1), nan(n,1), nan(n,1), nan(n,1), nan(n,1), ...
            zeros(n,1), zeros(n,1), nan(n,1), nan(n,1), ...
    'VariableNames', {'frame','valido','x_km','y_km','altura_km', ...
                      'guinada_deg','corroborados','hipotesis', ...
                      'error_pos_km','error_alt_km'});

prev = []; last = []; hueco = 0;
tic;

for i = 1:n
    f = frameIni + i - 1;
    Res.frame(i) = f;

    j = f - D.frame_min + 1;
    if j < 1 || j > numel(D.datos) || isempty(D.datos{j})
        hueco = hueco + 1;
        continue;
    end
    dets = D.datos{j};

    % prediccion a velocidad constante
    if isempty(last)
        pred = [];  radio = inf;
    elseif isempty(prev)
        pred = last;  radio = min(params.salto_max_km + hueco*params.margen_hueco, params.radio_max_km);
    else
        pred = last + (last - prev);
        radio = min(params.salto_max_km + hueco*params.margen_hueco, params.radio_max_km);
    end

    R = resolver_frame(dets, mapa, params, pred, radio);
    if isempty(R) && ~isempty(pred)
        R = resolver_frame(dets, mapa, params, [], inf);   % reintento sin prior
    end

    if isempty(R)
        hueco = hueco + 1;
        continue;
    end

    p = complex(R.x_km, R.y_km);

    % puerta de salto: un enganche erroneo no debe contaminar posPrev y
    % arrastrarse a los frames siguientes. Antes los fallos venian en
    % rachas contiguas justo por esto.
    if ~isempty(last) && abs(p - last) > radio
        hueco = hueco + 1;
        continue;
    end

    Res.valido(i)       = true;
    Res.x_km(i)         = R.x_km;
    Res.y_km(i)         = R.y_km;
    Res.altura_km(i)    = R.altura_km;
    Res.guinada_deg(i)  = R.guinada_deg;
    Res.corroborados(i) = R.corroborados;
    Res.hipotesis(i)    = R.n_hipotesis;

    if validar && isKey(real_, f)
        g = real_(f);
        Res.error_pos_km(i) = hypot(R.x_km - g(1), R.y_km - g(2));
        Res.error_alt_km(i) = R.altura_km - g(3);
    end

    prev = last; last = p; hueco = 0;

    if mod(i, 100) == 0
        fprintf('  %d/%d frames  (%.3f s/frame)\n', i, n, toc/i);
    end
end

%% --------------------------------------------------------------- resumen
ok = Res.valido;
fprintf('\n======================================================\n');
fprintf('Frames resueltos : %d de %d  (%.1f %%)\n', sum(ok), n, 100*sum(ok)/n);
fprintf('Tiempo           : %.3f s por frame\n', toc/n);
if validar
    e = Res.error_pos_km(ok);
    a = Res.error_alt_km(ok);
    fprintf('Error de posicion: mediana %.3f km   p95 %.3f km   RMS %.3f km\n', ...
        median(e), prctile_(e,95), sqrt(mean(e.^2)));
    fprintf('  aciertos < 5 km: %.1f %%\n', 100*mean(e < 5));
    fprintf('Error de altitud : sesgo %+.3f km   RMS %.3f km\n', mean(a), sqrt(mean(a.^2)));
    fprintf('Guinada estimada : media %+.4f deg   RMS %.4f deg\n', ...
        mean(Res.guinada_deg(ok)), sqrt(mean(Res.guinada_deg(ok).^2)));
end
fprintf('======================================================\n');

writetable(Res, rutaOut);
fprintf('Resultados en %s\n', rutaOut);

%% ----------------------------------------------------------------- plots
figure('Color','w','Name','Fase 3A','Position',[80 80 1500 460]);

subplot(1,3,1); hold on; grid on;
scatter(mapa.x, mapa.y, max(mapa.D*1.2,4), [.7 .7 .7], 'filled');
if validar
    plot(Treal.x_km, Treal.y_km, '-', 'Color', [.16 .43 .73], 'LineWidth', 2.2);
end
plot(Res.x_km(ok), Res.y_km(ok), '--', 'Color', [.82 .29 .36], 'LineWidth', 1.1);
axis equal; xlim([0 params.mapa_ancho_km]); ylim([0 params.mapa_alto_km]);
xlabel('x (km)'); ylabel('y (km)');
title(sprintf('Trayectoria  (%d/%d frames)', sum(ok), n));
legend({'mapa','real','estimada'}, 'Location','northwest', 'FontSize', 8);

subplot(1,3,2); grid on;
if validar
    semilogy(Res.frame(ok), Res.error_pos_km(ok), 'Color', [.82 .29 .36]);
    xlabel('frame'); ylabel('error de posicion (km)'); title('Error de posicion');
end

subplot(1,3,3); hold on; grid on;
if validar
    plot(Treal.frame, Treal.altura_km, 'Color', [.16 .43 .73], 'LineWidth', 2);
end
plot(Res.frame(ok), Res.altura_km(ok), 'Color', [.82 .29 .36]);
xlabel('frame'); ylabel('altitud (km)'); title('Altitud');
legend({'real','estimada'}, 'Location','best', 'FontSize', 8);


function p = prctile_(v, q)
% percentil sin el Statistics Toolbox
v = sort(v(:));
if isempty(v), p = NaN; return; end
i = max(1, min(numel(v), round(q/100 * numel(v))));
p = v(i);
end
