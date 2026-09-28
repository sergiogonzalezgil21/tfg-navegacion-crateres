% Figuras de resultados del capitulo 3. No calcula nada nuevo: lee los CSV que
% dejaron las fases y el detector, y guarda las cuatro figuras en latex\figs:
%
%   fig_error_frames.png   error de posicion a lo largo del vuelo (3A / 3B)
%   fig_altitud_fase4.png  cobertura y error por franja de altitud (fase 4)
%   fig_cascada.png        cobertura por altitud con los tres mapas
%   fig_rotacion.png       recall frente al giro, modelo original y reentrenado
%
% Conviene regenerarlas despues de repetir cualquier fase, para que las
% figuras y los numeros del capitulo salgan de la misma tanda.

aqui = fileparts(mfilename('fullpath'));   % carpeta matlab/ del repositorio
if isempty(aqui), aqui = pwd; end
raiz = fileparts(aqui);                    % raiz del repositorio
dirML  = fullfile(raiz, 'matlab');
dirIA  = fullfile(raiz, 'ia');
dirFig = fullfile(raiz, 'latex', 'figs');
if ~isfolder(dirFig), mkdir(dirFig); end

% Colores del documento (los mismos de estilo.tex)
AZUL  = [0 64 128]/255;
VERDE = [0 120 60]/255;
ROJO  = [170 30 30]/255;
GRIS  = [0.45 0.45 0.45];
% Rampa secuencial para la densidad del mapa: mas denso = mas oscuro
RAMPA = [0.62 0.74 0.86; 0.28 0.49 0.72; 0.06 0.22 0.42];

FS = 11;                                   % tamano de letra base

fprintf('Generando figuras del capitulo 3...\n');

%% ---------------------------------------------- FIGURA 1: error por frame
f3a = fullfile(dirML, 'fase3a_resultados.csv');
f3b = fullfile(dirML, 'fase3b_resultados.csv');
if isfile(f3a) && isfile(f3b)
    A = readtable(f3a);
    B = readtable(f3b);

    fig = figure('Color','w','Position',[100 100 1100 460]);
    ax  = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    set(ax,'YScale','log','FontSize',FS,'GridAlpha',0.12);

    okA = A.valido == 1 & ~isnan(A.error_pos_km);
    okB = B.valido == 1 & ~isnan(B.error_pos_km);

    plot(ax, A.frame(okA), A.error_pos_km(okA), '-', 'Color',[AZUL 0.85], ...
         'LineWidth',0.9, 'DisplayName','Ensayo 1 (diferido)');
    plot(ax, B.frame(okB), B.error_pos_km(okB), '-', 'Color',[VERDE 0.85], ...
         'LineWidth',0.9, 'DisplayName','Ensayo 2 (tiempo real)');

    % Fotogramas sin solucion, marcados abajo
    yFallo = 0.02;
    fA = A.frame(A.valido ~= 1);
    fB = B.frame(B.valido ~= 1);
    if ~isempty(fA)
        plot(ax, fA, yFallo*ones(size(fA)), 'v', 'MarkerSize',5, ...
             'MarkerFaceColor',AZUL, 'MarkerEdgeColor','none', ...
             'DisplayName','Sin solución (ensayo 1)');
    end
    if ~isempty(fB)
        plot(ax, fB, 1.5*yFallo*ones(size(fB)), 'v', 'MarkerSize',5, ...
             'MarkerFaceColor',VERDE, 'MarkerEdgeColor','none', ...
             'DisplayName','Sin solución (ensayo 2)');
    end

    xlabel(ax,'Fotograma');
    ylabel(ax,'Error de posición (km)');
    xlim(ax,[1 max(A.frame)]);
    ylim(ax,[0.015 10]);
    legend(ax,'Location','northwest','Box','off','FontSize',FS-1);

    exportgraphics(fig, fullfile(dirFig,'fig_error_frames.png'), 'Resolution',200);
    close(fig);
    fprintf('  fig_error_frames.png\n');
else
    warning('Faltan fase3a_resultados.csv o fase3b_resultados.csv');
end

%% ------------------------------------- FIGURA 2: cobertura frente a altitud
f4 = fullfile(dirML, 'fase4_directo_resultados.csv');
if isfile(f4)
    T = readtable(f4);
    bordes = 170:-10:50;                    % franjas de 10 km
    nB  = numel(bordes)-1;
    cen = zeros(nB,1); cob = nan(nB,1); err = nan(nB,1);
    for i = 1:nB
        alto = bordes(i); bajo = bordes(i+1);
        cen(i) = (alto+bajo)/2;
        m = T.altura_real_km < alto & T.altura_real_km >= bajo;
        if ~any(m), continue; end
        v = m & T.valido == 1;
        cob(i) = 100*sum(v)/sum(m);
        e = T.error_pos_km(v);
        e = e(~isnan(e) & e < 5);           % excluye los emparejamientos falsos
        if ~isempty(e), err(i) = median(e); end
    end

    fig = figure('Color','w','Position',[100 100 1000 440]);
    ax  = axes(fig); box(ax,'on'); set(ax,'FontSize',FS);

    yyaxis(ax,'left');
    bar(ax, cen, cob, 0.75, 'FaceColor',AZUL, 'EdgeColor','none', ...
        'FaceAlpha',0.85, 'DisplayName','Fotogramas resueltos');
    ylabel(ax,'Fotogramas resueltos (%)');
    ylim(ax,[0 105]); set(ax,'YColor',AZUL);

    yyaxis(ax,'right'); hold(ax,'on');
    plot(ax, cen, err, 'o-', 'Color',ROJO, 'MarkerFaceColor',ROJO, ...
         'LineWidth',1.6, 'MarkerSize',5, 'DisplayName','Error mediano');
    ylabel(ax,'Error mediano de posición (km)');
    ylim(ax,[0 0.8]); set(ax,'YColor',ROJO);

    xlabel(ax,'Altitud real (km)');
    set(ax,'XDir','reverse');               % el vuelo avanza hacia la izquierda
    xlim(ax,[min(bordes)-2 max(bordes)+2]);
    grid(ax,'on'); set(ax,'GridAlpha',0.12);
    legend(ax,'Location','southwest','Box','off','FontSize',FS-1);

    exportgraphics(fig, fullfile(dirFig,'fig_altitud_fase4.png'), 'Resolution',200);
    close(fig);
    fprintf('  fig_altitud_fase4.png\n');
else
    warning('Falta fase4_directo_resultados.csv');
end

%% ---------------------------------------------------- FIGURA 3: la cascada
fc = fullfile(dirML, 'fase4_cascada.csv');
if isfile(fc)
    T = readtable(fc);
    bordes = 170:-10:50;
    nB  = numel(bordes)-1;
    cen = zeros(nB,1); cob = nan(nB,3);
    cols = {'valido_D10','valido_D6','valido_D4'};
    for i = 1:nB
        alto = bordes(i); bajo = bordes(i+1);
        cen(i) = (alto+bajo)/2;
        m = T.altura_real_km < alto & T.altura_real_km >= bajo;
        if ~any(m), continue; end
        for j = 1:3
            cob(i,j) = 100*sum(m & T.(cols{j}) == 1)/sum(m);
        end
    end

    fig = figure('Color','w','Position',[100 100 1000 440]);
    ax  = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    set(ax,'FontSize',FS,'GridAlpha',0.12,'XDir','reverse');

    etiq = {'Mapa D \geq 10 km (258 cráteres)', ...
            'Mapa D \geq 6 km (440 cráteres)', ...
            'Mapa D \geq 4 km (753 cráteres)'};
    for j = 1:3
        plot(ax, cen, cob(:,j), 'o-', 'Color',RAMPA(j,:), ...
             'MarkerFaceColor',RAMPA(j,:), 'LineWidth',1.8, 'MarkerSize',5, ...
             'DisplayName',etiq{j});
    end

    xlabel(ax,'Altitud real (km)');
    ylabel(ax,'Fotogramas resueltos (%)');
    ylim(ax,[-3 105]);
    xlim(ax,[min(bordes)-2 max(bordes)+2]);
    legend(ax,'Location','southwest','Box','off','FontSize',FS-1);

    exportgraphics(fig, fullfile(dirFig,'fig_cascada.png'), 'Resolution',200);
    close(fig);
    fprintf('  fig_cascada.png\n');
else
    warning('Falta fase4_cascada.csv');
end

%% ------------------------------------------------- FIGURA 4: rotacion
fr = fullfile(dirIA, 'robustez_rotacion.csv');
if isfile(fr)
    T = readtable(fr);
    mod = string(T.modelo);
    esAntes = contains(mod, 'antes');

    fig = figure('Color','w','Position',[100 100 900 420]);
    ax  = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    set(ax,'FontSize',FS,'GridAlpha',0.12);

    plot(ax, T.giro(esAntes), 100*T.recall(esAntes), 'o--', 'Color',ROJO, ...
         'MarkerFaceColor',ROJO, 'LineWidth',1.8, 'MarkerSize',6, ...
         'DisplayName','Detector de crucero (sin rotación)');
    plot(ax, T.giro(~esAntes), 100*T.recall(~esAntes), 'o-', 'Color',VERDE, ...
         'MarkerFaceColor',VERDE, 'LineWidth',1.8, 'MarkerSize',6, ...
         'DisplayName','Detector reentrenado (\pm180\circ)');

    xlabel(ax,'Giro aplicado a la imagen (grados)');
    ylabel(ax,'Cráteres recuperados (%)');
    xticks(ax, unique(T.giro));
    ylim(ax,[-3 105]);
    legend(ax,'Location','southwest','Box','off','FontSize',FS-1);

    exportgraphics(fig, fullfile(dirFig,'fig_rotacion.png'), 'Resolution',200);
    close(fig);
    fprintf('  fig_rotacion.png\n');
else
    warning('Falta ia\\robustez_rotacion.csv');
end

fprintf('Hecho. Figuras en %s\n', dirFig);
