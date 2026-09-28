% Fase 4: descenso en directo sobre video.
%
% Igual que fase3b_directo, pero sobre el vuelo de descenso: la camara baja de
% 170 a 55 km girando +-180 grados y estabilizando la actitud para la
% aproximacion final.
%
% Dos cosas se ven aqui y no en la fase 3B. La camara gira, y el
% emparejamiento es invariante a rotacion: el panel de guinada sigue la
% sinusoide entera. Y el sistema pierde el enganche al bajar, en un punto
% concreto: por debajo de unos 118 km el detector encuentra ~3 crateres de
% mapa y hacen falta 4, asi que deja de dar solucion. No falla en silencio, se
% calla. Ese muro tiene dos causas multiplicandose, medidas en fase4_cascada:
% el area de la huella se encoge con h^2 y el detector pierde recall al bajar
% (72 % arriba, 52 % a 100 km) porque la textura pierde resolucion.
%
% El navegador es el mismo resolver_frame.m de las fases 3A y 3B; solo cambian
% los pesos del detector y dos parametros.
%
% Requiere haber ejecutado antes fase4_crear_video.

clc; close all;

%% ------------------------------------------------------------------ rutas
aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
rutaVid  = fullfile(raiz, 'blender', 'vuelo_descenso.mp4');
rutaMapa = fullfile(raiz, 'blender', 'mapa_referencia.csv');   % D >= 10 km,
                                    % el mismo de la fase 3, para que la
                                    % comparacion sea limpia. Con _D6 o _D4 se
                                    % ve el efecto de la densidad.
rutaReal = fullfile(raiz, 'blender', 'trayectoria_descenso.csv');
rutaCache= fullfile(raiz, 'matlab',  'tripletas_mapa.mat');
rutaOut  = fullfile(raiz, 'matlab',  'fase4_directo_resultados.csv');

% detector reentrenado con aumentacion por rotacion
pesos = fullfile(raiz, 'ia', 'runs', 'finetune_rotacion', 'weights', 'best.pt');

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
params.tol_forma = 0.02;
params.tol_diam  = 0.15;
params.altura_min_km = 40;      % junto con el tamano, es lo que fija la
params.altura_max_km = 200;     % escala: el descriptor no la fija
params.verif_px   = 55;
params.verif_diam = 0.30;
params.corrob_min = 4;
params.cluster_km = 5;
params.salto_max_km = 45;
params.margen_hueco = 12;
params.radio_max_km = 400;

% el detector de la fase 4 devuelve cajas infladas segun el giro (1.10 a
% 1.45): la verificacion compara tamanos relativos, acotados a ese rango
params.corregir_inflado = true;
params.inflado_min = 0.85;
params.inflado_max = 1.70;

CONF_YOLO = 0.70;
IMGSZ     = 1024;
PINTAR_CAJAS = true;
ESTELA = 90;

%% -------------------------------------------------------- arrancar Python
fprintf('Preparando el interprete de Python...\n');
pe = pyenv;
if ~isempty(PYTHON_EXE) && pe.Status == "Loaded" && ...
        ~strcmpi(char(pe.Executable), PYTHON_EXE)
    terminate(pyenv); pe = pyenv;
end
if pe.Status ~= "Loaded" || pe.ExecutionMode ~= "OutOfProcess"
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
yolo = py.importlib.import_module('detector_yolo');
py.importlib.reload(yolo);

fprintf('Cargando el detector (reentrenado con rotacion)...\n');
disp("  " + string(yolo.cargar(pesos, IMGSZ, CONF_YOLO)));
yolo.calentar(int32(params.frameHeight), int32(params.frameWidth), int32(2));

info = cell(yolo.abrir_video(rutaVid));
nFrames = double(info{1});
fprintf('Video: %d frames\n', nFrames);

%% ------------------------------------------------------------------ mapa
pm.lado_max_km = 200; pm.factor_lado = 0; pm.cache = rutaCache;
mapa = cargar_mapa(rutaMapa, pm);

Treal = readtable(rutaReal);
vr = VideoReader(rutaVid);

%% ---------------------------------------------------------------- figura
fig = figure('Color', [.10 .10 .12], 'Name', 'Fase 4 - descenso en directo', ...
    'Position', [30 30 1660 900], 'NumberTitle', 'off');
assignin('base','PARAR_4',false);
uicontrol(fig, 'Style','pushbutton', 'String','PARAR', ...
    'Units','normalized', 'Position',[0.905 0.955 0.08 0.038], ...
    'FontWeight','bold', 'Callback', @(~,~) assignin('base','PARAR_4',true));

col = struct('real',[.35 .60 .88], 'est',[.95 .30 .32], ...
             'caja',[.55 .55 .60], 'cajaUsa',[1 .85 .15], 'txt',[.92 .92 .92]);

axV = axes('Parent',fig,'Position',[0.015 0.27 0.52 0.69]);
hImg = image(axV, zeros(params.frameHeight, params.frameWidth, 3, 'uint8'));
axis(axV,'image'); set(axV,'XTick',[],'YTick',[]); hold(axV,'on');
hCaja  = plot(axV, nan, nan, '-', 'Color', col.caja,    'LineWidth', 0.8);
hCajaU = plot(axV, nan, nan, '-', 'Color', col.cajaUsa, 'LineWidth', 2.0);
hTitV = title(axV, 'Camara', 'Color', col.txt);
hTxt = text(axV, 20, params.frameHeight-20, '', 'Color', col.txt, ...
    'FontName','Consolas', 'FontSize', 11, 'VerticalAlignment','bottom', ...
    'BackgroundColor',[0 0 0 0.55], 'Margin', 6, 'Interpreter','none');

axM = axes('Parent',fig,'Position',[0.565 0.63 0.42 0.34],'Color',[.14 .14 .17]);
hold(axM,'on');
scatter(axM, mapa.x, mapa.y, max(mapa.D*0.9,3), [.45 .45 .5], 'filled');
plot(axM, Treal.x_km, Treal.y_km, '-', 'Color', [.30 .42 .58], 'LineWidth', 1.2);
hTray = plot(axM, nan, nan, '-', 'Color', col.est, 'LineWidth', 1.6);
hUsad = plot(axM, nan, nan, 'o', 'MarkerSize', 8, 'Color', col.cajaUsa, 'LineWidth', 1.4);
hNave = plot(axM, nan, nan, 'p', 'MarkerSize', 15, 'MarkerFaceColor', col.est, ...
    'MarkerEdgeColor','w');
xlim(axM,[0 params.mapa_ancho_km]); ylim(axM,[0 params.mapa_alto_km]);
pbaspect(axM,[params.mapa_ancho_km params.mapa_alto_km 1]);
set(axM,'XColor',col.txt,'YColor',col.txt,'GridColor',[.4 .4 .45]); grid(axM,'on');
title(axM,'Mapa embarcado','Color',col.txt);

axA = axes('Parent',fig,'Position',[0.565 0.38 0.42 0.19],'Color',[.14 .14 .17]);
hold(axA,'on');
plot(axA, Treal.frame, Treal.altura_km, '-', 'Color', col.real, 'LineWidth', 1.8);
hAlt = plot(axA, nan, nan, '.', 'Color', col.est, 'MarkerSize', 6);
yline(axA, 118, '--', '118 km', 'Color', [.9 .5 .2], 'LabelHorizontalAlignment','left');
xlim(axA,[1 nFrames]); ylim(axA,[40 200]);
set(axA,'XColor',col.txt,'YColor',col.txt,'GridColor',[.4 .4 .45]); grid(axA,'on');
ylabel(axA,'altitud (km)','Color',col.txt);

axG = axes('Parent',fig,'Position',[0.565 0.14 0.42 0.19],'Color',[.14 .14 .17]);
hold(axG,'on');
plot(axG, Treal.frame, Treal.rot_z_deg, '-', 'Color', col.real, 'LineWidth', 1.8);
hYaw = plot(axG, nan, nan, '.', 'Color', col.est, 'MarkerSize', 6);
xlim(axG,[1 nFrames]); ylim(axG,[-200 200]);
set(axG,'XColor',col.txt,'YColor',col.txt,'GridColor',[.4 .4 .45]); grid(axG,'on');
ylabel(axG,'guinada (deg)','Color',col.txt); xlabel(axG,'frame','Color',col.txt);

axE = axes('Parent',fig,'Position',[0.015 0.05 0.52 0.17],'Color',[.14 .14 .17]);
hold(axE,'on');
hErr = plot(axE, nan, nan, '.', 'Color', col.est, 'MarkerSize', 6);
xlim(axE,[1 nFrames]); set(axE,'YScale','log'); ylim(axE,[1e-2 1e3]);
set(axE,'XColor',col.txt,'YColor',col.txt,'GridColor',[.4 .4 .45]); grid(axE,'on');
xlabel(axE,'frame','Color',col.txt); ylabel(axE,'error (km)','Color',col.txt);

%% ------------------------------------------------------------ bucle vivo
X=nan(nFrames,1); Y=nan(nFrames,1); A=nan(nFrames,1);
G=nan(nFrames,1); C=nan(nFrames,1); E=nan(nFrames,1); V=false(nFrames,1);
prev=[]; last=[]; hueco=0; kUlt=0; altPerdida=NaN;
t0=tic; tUlt=0; fps=0;

for k = 1:nFrames
    if ~ishandle(fig) || evalin('base','PARAR_4'), break; end

    res = cell(yolo.siguiente());
    nF = double(res{1});
    if nF == 0, break; end
    raw = uint8(res{2});
    if isempty(raw), Dt = zeros(0,5);
    else, Dt = reshape(typecast(raw,'double'), 5, []).'; end

    if isempty(Dt)
        dets = zeros(0,4);
    else
        b = Dt(:,1)-Dt(:,3)/2 < 1 | Dt(:,1)+Dt(:,3)/2 > params.frameWidth-1 | ...
            Dt(:,2)-Dt(:,4)/2 < 1 | Dt(:,2)+Dt(:,4)/2 > params.frameHeight-1;
        dets = [Dt(:,1), Dt(:,2), Dt(:,3), double(b)];
    end

    if isempty(last)
        pred = []; radio = inf;
    else
        if isempty(prev), pred = last; else, pred = last + (last-prev); end
        radio = min(params.salto_max_km + hueco*params.margen_hueco, params.radio_max_km);
    end
    R = resolver_frame(dets, mapa, params, pred, radio);
    if isempty(R) && ~isempty(pred)
        R = resolver_frame(dets, mapa, params, [], inf);
    end
    if ~isempty(R)
        p = complex(R.x_km, R.y_km);
        if ~isempty(last) && abs(p-last) > radio, R = []; end
    end

    if isempty(R)
        hueco = hueco + 1;
        if hueco == 1 && k > 1 && isnan(altPerdida) && any(V(1:k-1))
            altPerdida = Treal.altura_km(k);   % altura del primer fallo
        end
    else
        V(k)=true; X(k)=R.x_km; Y(k)=R.y_km; A(k)=R.altura_km;
        G(k)=R.guinada_deg; C(k)=R.corroborados;
        E(k)=hypot(R.x_km-Treal.x_km(k), R.y_km-Treal.y_km(k));
        prev=last; last=complex(R.x_km,R.y_km); hueco=0;
    end

    if hasFrame(vr), set(hImg,'CData',readFrame(vr)); end

    if PINTAR_CAJAS && ~isempty(Dt)
        [~,o]=sort(Dt(:,3),'descend'); nU=min(params.K,size(Dt,1));
        aC=cajas(Dt(o(nU+1:end),:)); set(hCaja,aC{:});
        aU=cajas(Dt(o(1:nU),:));     set(hCajaU,aU{:});
    end

    ini=max(1,k-ESTELA);
    set(hTray,'XData',X(ini:k),'YData',Y(ini:k));
    set(hAlt,'XData',1:k,'YData',A(1:k));
    set(hYaw,'XData',1:k,'YData',G(1:k));
    set(hErr,'XData',1:k,'YData',E(1:k));
    if V(k)
        set(hNave,'XData',X(k),'YData',Y(k));
        set(hUsad,'XData',mapa.x(R.ids_mapa),'YData',mapa.y(R.ids_mapa));
    end

    t=toc(t0);
    if t-tUlt > 0.5, fps = k/t; tUlt = t; end

    if V(k)
        txt = sprintf(['frame     %4d / %d\n' ...
                       'altitud   %6.1f km  (real %6.1f)\n' ...
                       'guinada   %+7.1f deg  (real %+7.1f)\n' ...
                       'crateres  %d corroborados de %d cajas\n' ...
                       'error     %7.3f km\n%5.2f fps'], ...
            k, nFrames, A(k), Treal.altura_km(k), G(k), Treal.rot_z_deg(k), ...
            C(k), size(Dt,1), E(k), fps);
        set(hTxt,'Color',col.txt);
    else
        txt = sprintf(['frame     %4d / %d\n' ...
                       'SIN SOLUCION   (%d frames seguidos)\n' ...
                       'altitud real %6.1f km\n' ...
                       'cajas     %d\n%5.2f fps'], ...
            k, nFrames, hueco, Treal.altura_km(k), size(Dt,1), fps);
        set(hTxt,'Color',[1 .45 .35]);
        set(hUsad,'XData',nan,'YData',nan);
    end
    set(hTxt,'String',txt);
    set(hTitV,'String',sprintf('Camara - frame %d', k));
    kUlt = k;
    drawnow limitrate;
end

yolo.cerrar();

%% --------------------------------------------------------------- resumen
n = kUlt;
if n < 1, fprintf('No se proceso ningun frame.\n'); return; end
ok = V(1:n);
e = E(1:n); e = e(ok & ~isnan(e));
fprintf('\n======================================================\n');
fprintf('Frames procesados : %d\n', n);
fprintf('Con solucion      : %d  (%.1f %%)\n', sum(ok), 100*mean(ok));
fprintf('Correctos (<5 km) : %d  (%.1f %%)\n', sum(e<5), 100*mean(e<5));
if any(e<5)
    fprintf('Error de posicion : mediana %.3f km   (solo los correctos)\n', ...
        median(e(e<5)));
end
if ~isnan(altPerdida)
    fprintf('Primer fallo a    : %.1f km de altitud\n', altPerdida);
end
fprintf('Ritmo             : %.2f fps\n', fps);
fprintf('======================================================\n');

T = table((1:n)', V(1:n), Treal.altura_km(1:n), Treal.rot_z_deg(1:n), ...
    X(1:n), Y(1:n), A(1:n), G(1:n), C(1:n), E(1:n), ...
    'VariableNames', {'frame','valido','altura_real_km','guinada_real_deg', ...
                      'x_km','y_km','altura_km','guinada_deg', ...
                      'corroborados','error_pos_km'});
writetable(T, rutaOut);
fprintf('Resultados en %s\n', rutaOut);


% =========================================================================
function args = cajas(D)
% Polilinea con NaN entre rectangulos: un solo objeto grafico en vez de N.
if isempty(D), args = {'XData', nan, 'YData', nan}; return; end
x0=D(:,1)-D(:,3)/2; x1=D(:,1)+D(:,3)/2;
y0=D(:,2)-D(:,4)/2; y1=D(:,2)+D(:,4)/2;
nc=nan(size(x0));
X=[x0 x1 x1 x0 x0 nc].'; Y=[y0 y0 y1 y1 y0 nc].';
args={'XData',X(:),'YData',Y(:)};
end
