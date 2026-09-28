% Fase 3B: navegacion en directo sobre video.
% TFG - Navegacion optica relativa al terreno mediante lectura de crateres.
%
% Reproduce el video del vuelo y, frame a frame y en vivo, YOLO detecta los
% crateres, se pintan las cajas sobre la imagen, el navegador empareja contra
% el mapa embarcado y estima la pose, y las graficas se van dibujando solas.
% No hay pausas artificiales: va al ritmo del detector, unos 4 frames/s.
%
% MATLAB dirige: llama a Python por la interfaz py.* para la deteccion y usa
% el mismo resolver_frame.m validado en la fase 3A, sin cambiar una linea. Lo
% unico distinto entre 3A y 3B es de donde vienen las detecciones.
%
% El video lo lee Python, no MATLAB. Pasar cada frame de 1920x1080 por la
% frontera son 6.2 MB por llamada, y con el interprete fuera de proceso eso se
% serializa entero. Python abre el video y lee secuencialmente; MATLAB abre el
% mismo fichero solo para mostrarlo, y cada llamada devuelve el numero de
% frame para comprobar que no se han desincronizado.
%
% Requiere haber ejecutado antes fase3b_crear_video, que monta
% blender\vuelo.mp4. Si el python por defecto no tiene ultralytics, hay que
% fijar PYTHON_EXE mas abajo.

clc; close all;

%% ------------------------------------------------------------------ rutas
aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
rutaVid  = fullfile(raiz, 'blender', 'vuelo.mp4');
rutaMapa = fullfile(raiz, 'blender', 'mapa_referencia.csv');
rutaReal = fullfile(raiz, 'blender', 'trayectoria_real.csv');   % solo validar
rutaCache= fullfile(raiz, 'matlab',  'tripletas_mapa.mat');
rutaOut  = fullfile(raiz, 'matlab',  'fase3b_resultados.csv');

pesos = fullfile(raiz, 'ia', 'runs', 'finetune_blender', 'weights', 'best.pt');

% Interprete de Python con ultralytics instalado. Dejandolo vacio se usa el
% que MATLAB ya tenga configurado, que se consulta con  pyenv  desde la
% linea de comandos. Para fijar otro, poner aqui la ruta a su python.exe.
PYTHON_EXE = '';

%% ------------------------------------------------------------- parametros
params.frameWidth  = 1920;
params.frameHeight = 1080;
params.FOV_horizontal = 60;
params.mapa_ancho_km = 840;
params.mapa_alto_km  = 630;

params.K = 7;
params.lado_max_km = 200;
params.tol_forma = 0.02;
params.tol_diam  = 0.15;
params.altura_min_km = 50;
params.altura_max_km = 400;
params.verif_px   = 55;
params.verif_diam = 0.30;
params.corrob_min = 4;
params.cluster_km = 5;
params.salto_max_km = 45;
params.margen_hueco = 12;
params.radio_max_km = 400;

CONF_YOLO   = 0.70;    % punto de operacion validado en la fase 2
IMGSZ       = 1024;
PINTAR_CAJAS = true;   % recuadros de la IA sobre el video
ESTELA       = 90;     % frames de estela de la trayectoria estimada

%% -------------------------------------------------------- arrancar Python
fprintf('Preparando el interprete de Python...\n');
pe = pyenv;
if ~isempty(PYTHON_EXE) && pe.Status == "Loaded" && ...
        ~strcmpi(char(pe.Executable), PYTHON_EXE)
    terminate(pyenv);        % no se puede cambiar el interprete en caliente
    pe = pyenv;
end
if pe.Status ~= "Loaded" || pe.ExecutionMode ~= "OutOfProcess"
    % OutOfProcess a proposito: torch y numpy cargan librerias nativas que
    % chocan con las de MATLAB. En InProcess eso puede tirar MATLAB entero
    % sin ningun aviso.
    if isempty(PYTHON_EXE)
        pyenv('ExecutionMode', 'OutOfProcess');
    else
        pyenv('Version', PYTHON_EXE, 'ExecutionMode', 'OutOfProcess');
    end
end

carpeta = fileparts(mfilename('fullpath'));
if count(py.sys.path, carpeta) == 0
    insert(py.sys.path, int32(0), carpeta);
end
% ojo: no llamarlo 'det', que es el determinante de MATLAB
yolo = py.importlib.import_module('detector_yolo');
py.importlib.reload(yolo);

fprintf('Cargando el detector...\n');
disp("  " + string(yolo.cargar(pesos, IMGSZ, CONF_YOLO)));
fprintf('Calentando (para que el primer frame del directo no se atasque)...\n');
yolo.calentar(int32(params.frameHeight), int32(params.frameWidth), int32(2));

info = cell(yolo.abrir_video(rutaVid));
nFrames = double(info{1});
fprintf('Video: %d frames, %dx%d\n', nFrames, double(info{2}), double(info{3}));

%% ------------------------------------------------------------------ mapa
pm.lado_max_km = params.lado_max_km;
pm.cache = rutaCache;
mapa = cargar_mapa(rutaMapa, pm);

validar = isfile(rutaReal);
if validar
    Treal = readtable(rutaReal);
end

vr = VideoReader(rutaVid);   % solo para mostrar

%% ---------------------------------------------------------------- figura
fig = figure('Color', [.10 .10 .12], 'Name', 'Fase 3B - navegacion en directo', ...
    'Position', [40 40 1620 880], 'NumberTitle', 'off');

parar = false;
uicontrol(fig, 'Style','pushbutton', 'String','PARAR', ...
    'Units','normalized', 'Position',[0.905 0.955 0.08 0.038], ...
    'FontWeight','bold', 'Callback', @(~,~) assignin('base','PARAR_3B',true));
assignin('base','PARAR_3B',false);

col = struct('real',[.35 .60 .88], 'est',[.95 .30 .32], ...
             'caja',[.20 .95 .55], 'cajaUsa',[1 .85 .15], 'txt',[.92 .92 .92]);

% --- video
axV = axes('Parent',fig, 'Position',[0.015 0.30 0.545 0.66]);
hImg = image(axV, zeros(params.frameHeight, params.frameWidth, 3, 'uint8'));
axis(axV, 'image'); set(axV, 'XTick', [], 'YTick', []);
hold(axV,'on');
hCaja  = plot(axV, nan, nan, '-', 'Color', col.caja,    'LineWidth', 1.0);
hCajaU = plot(axV, nan, nan, '-', 'Color', col.cajaUsa, 'LineWidth', 2.0);
hTitV = title(axV, 'Camara', 'Color', col.txt);

% --- mapa
axM = axes('Parent',fig, 'Position',[0.605 0.56 0.375 0.40], 'Color',[.14 .14 .17]);
hold(axM,'on');
scatter(axM, mapa.x, mapa.y, max(mapa.D*0.9,3), [.45 .45 .5], 'filled');
if validar
    plot(axM, Treal.x_km, Treal.y_km, '-', 'Color', [.30 .42 .58], 'LineWidth', 1.2);
end
hTray = plot(axM, nan, nan, '-', 'Color', col.est, 'LineWidth', 1.6);
hUsad = plot(axM, nan, nan, 'o', 'MarkerSize', 8, 'Color', col.cajaUsa, 'LineWidth', 1.4);
hNave = plot(axM, nan, nan, 'p', 'MarkerSize', 16, ...
    'MarkerFaceColor', col.est, 'MarkerEdgeColor','w');
xlim(axM,[0 params.mapa_ancho_km]); ylim(axM,[0 params.mapa_alto_km]);
pbaspect(axM, [params.mapa_ancho_km params.mapa_alto_km 1]);   % no 'axis equal'
set(axM,'XColor',col.txt,'YColor',col.txt,'GridColor',[.4 .4 .45]); grid(axM,'on');
title(axM,'Mapa embarcado','Color',col.txt);

% --- altitud
axA = axes('Parent',fig, 'Position',[0.605 0.31 0.375 0.175], 'Color',[.14 .14 .17]);
hold(axA,'on');
if validar
    plot(axA, Treal.frame, Treal.altura_km, '-', 'Color', col.real, 'LineWidth', 1.6);
end
hAlt = plot(axA, nan, nan, '-', 'Color', col.est, 'LineWidth', 1.2);
xlim(axA,[1 nFrames]); ylim(axA,[100 190]);
set(axA,'XColor',col.txt,'YColor',col.txt,'GridColor',[.4 .4 .45]); grid(axA,'on');
ylabel(axA,'altitud (km)','Color',col.txt);
title(axA,'Altitud','Color',col.txt);

% --- error
axE = axes('Parent',fig, 'Position',[0.015 0.065 0.545 0.175], 'Color',[.14 .14 .17]);
hold(axE,'on');
hErr = plot(axE, nan, nan, '-', 'Color', col.est, 'LineWidth', 1.0);
xlim(axE,[1 nFrames]); set(axE,'YScale','log'); ylim(axE,[1e-2 1e2]);
set(axE,'XColor',col.txt,'YColor',col.txt,'GridColor',[.4 .4 .45]); grid(axE,'on');
xlabel(axE,'frame','Color',col.txt); ylabel(axE,'error (km)','Color',col.txt);
title(axE,'Error de posicion contra la verdad','Color',col.txt);

% --- marcador
axT = axes('Parent',fig,'Position',[0.605 0.04 0.375 0.22],'Visible','off');
hTxt = text(axT, 0, 1, '', 'Units','normalized', 'VerticalAlignment','top', ...
    'FontName','Consolas', 'FontSize', 11.5, 'Color', col.txt, 'Interpreter','none');

%% ------------------------------------------------------------ bucle vivo
X = nan(nFrames,1); Y = nan(nFrames,1); A = nan(nFrames,1);
G = nan(nFrames,1); C = nan(nFrames,1); E = nan(nFrames,1);
V = false(nFrames,1);

prev = []; last = []; hueco = 0; kUlt = 0;
t0 = tic; tUlt = 0; fps = 0;

for k = 1:nFrames
    if ~ishandle(fig) || evalin('base','PARAR_3B'), break; end

    % ---- deteccion en vivo ----
    res = cell(yolo.siguiente());
    nF  = double(res{1});
    if nF == 0, break; end
    if nF ~= k
        warning('Desincronizado: Python va por el frame %d y MATLAB por el %d.', nF, k);
    end

    raw = uint8(res{2});
    if isempty(raw)
        Dt = zeros(0,5);
    else
        Dt = reshape(typecast(raw,'double'), 5, []).';   % cx cy w h conf
    end

    % ---- a formato del navegador: [cx cy w borde] ----
    if isempty(Dt)
        dets = zeros(0,4);
    else
        borde = Dt(:,1)-Dt(:,3)/2 < 1 | Dt(:,1)+Dt(:,3)/2 > params.frameWidth-1 | ...
                Dt(:,2)-Dt(:,4)/2 < 1 | Dt(:,2)+Dt(:,4)/2 > params.frameHeight-1;
        dets = [Dt(:,1), Dt(:,2), Dt(:,3), double(borde)];
    end

    % ---- navegacion (el MISMO resolver de la fase 3A) ----
    if isempty(last)
        pred = []; radio = inf;
    else
        if isempty(prev), pred = last; else, pred = last + (last - prev); end
        radio = min(params.salto_max_km + hueco*params.margen_hueco, params.radio_max_km);
    end
    R = resolver_frame(dets, mapa, params, pred, radio);
    if isempty(R) && ~isempty(pred)
        R = resolver_frame(dets, mapa, params, [], inf);
    end
    if ~isempty(R)
        p = complex(R.x_km, R.y_km);
        if ~isempty(last) && abs(p - last) > radio
            R = [];                       % puerta de salto
        end
    end

    if isempty(R)
        hueco = hueco + 1;
    else
        V(k)=true; X(k)=R.x_km; Y(k)=R.y_km; A(k)=R.altura_km;
        G(k)=R.guinada_deg; C(k)=R.corroborados;
        if validar && k <= height(Treal)
            E(k) = hypot(R.x_km - Treal.x_km(k), R.y_km - Treal.y_km(k));
        end
        prev = last; last = complex(R.x_km, R.y_km); hueco = 0;
    end

    % ---- video ----
    if hasFrame(vr)
        img = readFrame(vr);
        set(hImg, 'CData', img);
    end

    if PINTAR_CAJAS && ~isempty(Dt)
        [~, ord] = sort(Dt(:,3), 'descend');
        nUsa = min(params.K, size(Dt,1));
        aC = cajas(Dt(ord(nUsa+1:end),:));  set(hCaja,  aC{:});
        aU = cajas(Dt(ord(1:nUsa),:));      set(hCajaU, aU{:});
    else
        set(hCaja,'XData',nan,'YData',nan); set(hCajaU,'XData',nan,'YData',nan);
    end

    % ---- graficas ----
    ini = max(1, k-ESTELA);
    set(hTray, 'XData', X(ini:k), 'YData', Y(ini:k));
    set(hAlt,  'XData', 1:k, 'YData', A(1:k));
    set(hErr,  'XData', 1:k, 'YData', E(1:k));
    if V(k)
        set(hNave, 'XData', X(k), 'YData', Y(k));
        set(hUsad, 'XData', mapa.x(R.ids_mapa), 'YData', mapa.y(R.ids_mapa));
    end

    t = toc(t0);
    if t - tUlt > 0.5, fps = k / t; tUlt = t; end

    if V(k)
        est = sprintf(['frame      %4d / %d\n' ...
                       'posicion   %7.2f , %7.2f km\n' ...
                       'altitud    %7.2f km\n' ...
                       'guinada    %7.2f deg\n' ...
                       'crateres   %d corroborados de %d cajas\n' ...
                       'error      %7.3f km\n' ...
                       'ritmo      %5.2f fps   (%.0f s)'], ...
            k, nFrames, X(k), Y(k), A(k), G(k), C(k), size(Dt,1), E(k), fps, t);
    else
        est = sprintf(['frame      %4d / %d\n' ...
                       'SIN SOLUCION  (%d frames seguidos)\n' ...
                       'cajas      %d\n' ...
                       'ritmo      %5.2f fps   (%.0f s)'], ...
            k, nFrames, hueco, size(Dt,1), fps, t);
    end
    set(hTxt, 'String', est);
    set(hTitV, 'String', sprintf('Camara - frame %d', k));

    kUlt = k;
    drawnow limitrate;
end

yolo.cerrar();

%% --------------------------------------------------------------- resumen
n = kUlt;
if n < 1, fprintf('No se proceso ningun frame.\n'); return; end
ok = V(1:n);
fprintf('\n======================================================\n');
fprintf('Frames procesados : %d\n', max(n,0));
fprintf('Resueltos         : %d  (%.1f %%)\n', sum(ok), 100*mean(ok));
fprintf('Ritmo medio       : %.2f fps\n', fps);
if validar && any(ok)
    e = E(1:n); e = e(ok & ~isnan(e));
    fprintf('Error de posicion : mediana %.3f km   RMS %.3f km\n', ...
        median(e), sqrt(mean(e.^2)));
end
fprintf('======================================================\n');

if n > 0
    T = table((1:n)', V(1:n), X(1:n), Y(1:n), A(1:n), G(1:n), C(1:n), E(1:n), ...
        'VariableNames', {'frame','valido','x_km','y_km','altura_km', ...
                          'guinada_deg','corroborados','error_pos_km'});
    writetable(T, rutaOut);
    fprintf('Resultados en %s\n', rutaOut);
end


% =========================================================================
function args = cajas(D)
% Convierte las cajas en una polilinea con NaN entre medias. Asi todas se
% dibujan con UN solo objeto grafico en vez de N rectangulos, que es lo que
% permite que el refresco siga el ritmo del video.
if isempty(D)
    args = {'XData', nan, 'YData', nan};
    return;
end
x0 = D(:,1)-D(:,3)/2;  x1 = D(:,1)+D(:,3)/2;
y0 = D(:,2)-D(:,4)/2;  y1 = D(:,2)+D(:,4)/2;
nanc = nan(size(x0));
X = [x0 x1 x1 x0 x0 nanc].';
Y = [y0 y0 y1 y1 y0 nanc].';
args = {'XData', X(:), 'YData', Y(:)};
end
