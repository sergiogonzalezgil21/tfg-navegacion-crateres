function R = resolver_frame(dets, mapa, params, pred, radio)
% RESOLVER_FRAME  Posicion, altitud y guinada a partir de las detecciones de
% un frame, sin conocer nada de la pose de antemano.
%
% Entradas
%   dets   : Nx4 [cx_px, cy_px, w_px, borde]
%   mapa   : struct de cargar_mapa
%   params : parametros
%   pred   : posicion predicha (complejo x+iy en km) o [] si no hay
%   radio  : km alrededor de pred donde se admiten hipotesis
%
% Salida, o [] si el frame no se resuelve
%   R.x_km, R.y_km, R.altura_km, R.guinada_deg, R.corroborados,
%   R.n_hipotesis, R.ids_mapa, R.idx_det
%
% Procedimiento:
%   1. Se cogen las K detecciones mas grandes. Por construccion del terreno
%      los crateres de escala de mapa son los mayores de la escena, y el
%      criterio es invariante a escala, lo que importa al no conocer aun la
%      altitud.
%   2. Cada tripleta de detecciones da un descriptor de 4 numeros que se busca
%      en la tabla del mapa.
%   3. Cada casamiento define una transformacion imagen->mapa completa
%      (escala, rotacion y traslacion) por minimos cuadrados.
%   4. Verificacion: se proyectan todas las detecciones con esa transformacion
%      y se cuenta cuantas caen sobre un crater del mapa del tamano que les
%      toca. Una hipotesis correcta corrobora 5 o 6; una casual, las 3 que la
%      generaron y poco mas. Sin la comprobacion de tamano no discrimina: se
%      midio una hipotesis falsa ganando a la correcta por un crater.
%   5. La ganadora se reajusta con todas las parejas corroboradas.
%
% La escala absoluta la fijan esa comprobacion de tamano y el rango
% params.altura_min_km / altura_max_km, que por eso conviene que sea estrecho
% y realista. El descriptor es invariante a escala a proposito, asi que un
% triangulo del mapa que fuese una copia al doble casa igual de bien; con una
% comprobacion puramente relativa el cociente D_det*ka/D_map sale identico y
% nada impide aceptarla. Medido: soluciones a 200-400 km de altitud con
% errores de posicion de 100 a 1000 km.
%
% params.inflado_min / inflado_max  rango plausible del factor de inflado de
% las cajas. params.corregir_inflado, false por defecto, activa la
% comprobacion de tamano invariante a escala del paso 4; hace falta con el
% detector reentrenado de la fase 4, que devuelve cajas infladas segun el
% giro, y se deja apagado para que 3A y 3B den los numeros documentados.

R = [];

if nargin < 4, pred = []; end
if nargin < 5, radio = inf; end

if size(dets,1) < 3, return; end

% ---- las K mas grandes ----
[~, o] = sort(dets(:,3), 'descend');
dets = dets(o(1:min(params.K, size(dets,1))), :);
nd = size(dets,1);
if nd < 3, return; end

s_det  = complex(dets(:,1), -dets(:,2));   % a marco y-arriba
d_det  = dets(:,3);
noBorde = dets(:,4) < 0.5;

% ---- hipotesis ----
C = nchoosek(1:nd, 3);
A = []; MS = []; MD_ = [];

for t = 1:size(C,1)
    ii = C(t,:);
    p  = [real(s_det(ii)), imag(s_det(ii))];
    L  = [ norm(p(2,:)-p(3,:)), norm(p(1,:)-p(3,:)), norm(p(1,:)-p(2,:)) ];
    [Ls, ord] = sort(L);
    if Ls(3) <= 0, continue; end
    ij = ii(ord);
    dd = d_det(ij);
    if dd(3) <= 0, continue; end

    q = [Ls(1)/Ls(3), Ls(2)/Ls(3), dd(1)/dd(3), dd(2)/dd(3)];
    cand = buscar_tripletas(mapa.trip, q, params.tol_forma, params.tol_diam);
    if isempty(cand), continue; end

    src = s_det(ij);                       % 3x1
    ms  = mean(src);
    sc  = src - ms;
    den = sum(abs(sc).^2);
    if den <= 0, continue; end

    % OJO con el indexado: ids(cand,:) es Mx3, pero cuando M = 1 MATLAB lo
    % trata como VECTOR y el resultado toma la orientacion de mapa.xy, que
    % es columna: saldria 3x1 en vez de 1x3. El reshape lo fuerza siempre.
    dst = reshape(mapa.xy(mapa.trip.ids(cand,:)), [], 3);   % Mx3
    md  = mean(dst, 2);
    dc  = dst - md;

    a = (dc * conj(sc)) / den;             % Mx1, escala y rotacion

    A    = [A;    a];      %#ok<AGROW>
    MS   = [MS;   repmat(ms, numel(a), 1)]; %#ok<AGROW>
    MD_  = [MD_;  md];     %#ok<AGROW>
end

if isempty(A), return; end

% ---- filtros fisicos ----
centro = complex(params.frameWidth/2, -params.frameHeight/2);
pos = A .* (centro - MS) + MD_;
ka  = abs(A);
alt = params.frameWidth * ka / (2*tand(params.FOV_horizontal/2));

ok = isfinite(A) & ka > 0 ...
   & alt > params.altura_min_km & alt < params.altura_max_km ...
   & real(pos) > 0 & real(pos) < params.mapa_ancho_km ...
   & imag(pos) > 0 & imag(pos) < params.mapa_alto_km;

if ~isempty(pred) && isfinite(radio)
    ok = ok & abs(pos - pred) < radio;
end
if ~any(ok), return; end

A = A(ok); MS = MS(ok); MD_ = MD_(ok); pos = pos(ok); ka = ka(ok);
nh = numel(A);

% ---- verificacion contra todo el mapa, por trozos ----
corrob = zeros(nh,1);
VAL    = false(nh, nd);
IMIN   = zeros(nh, nd);
paso   = max(1, floor(2e6 / max(nd*mapa.n,1)));

for i0 = 1:paso:nh
    i1 = min(i0+paso-1, nh);
    r  = (i0:i1).';
    proj = A(r) .* (s_det.' - MS(r)) + MD_(r);      % (nr x nd)
    dif  = abs(reshape(proj, [], 1) - mapa.xy.');   % (nr*nd x n)
    [dm, im] = min(dif, [], 2);
    dm = reshape(dm, numel(r), nd);
    im = reshape(im, numel(r), nd);

    cerca = dm < params.verif_px * ka(r);
    % el crater del mapa tiene que TENER EL TAMANO que le toca.
    % Mismo cuidado con el indexado: con una sola hipotesis mapa.D(im)
    % saldria nd x 1 en vez de 1 x nd y la resta se expandiria a nd x nd
    % sin dar ningun error.
    Dmap = reshape(mapa.D(im), size(im));
    esperado = d_det.' .* ka(r);

    if isfield(params, 'corregir_inflado') && params.corregir_inflado
        % ---------------------------------------------------------------
        % COMPROBACION DE TAMANO INVARIANTE A ESCALA
        %
        % El detector reentrenado con degrees=180 devuelve cajas INFLADAS,
        % y el factor depende del giro de la imagen: se midio 1.16 a 0
        % grados y 1.43 a 45. Sigue |cos|+|sin| con correlacion 0.99, que
        % es exactamente cuanto crece un rectangulo alineado con los ejes
        % al envolver un cuadrado girado. La red aprendio a reproducir el
        % inflado de la aumentacion leyendo la direccion de la sombra.
        %
        % Comparar diametros absolutos con una tolerancia del 30 % no
        % sobrevive a eso. Pero dentro de un frame TODOS los crateres
        % comparten el mismo giro y por tanto el mismo factor, asi que en
        % vez del tamano absoluto se exige que los tamanos sean
        % CONSISTENTES ENTRE SI: se estima el factor como la mediana de la
        % propia hipotesis y se comprueba la dispersion alrededor de ella.
        %
        % Con un detector sin inflar el factor sale ~1 y la comprobacion
        % es equivalente a la de antes.
        % ---------------------------------------------------------------
        rho = esperado ./ max(Dmap, 1e-9);
        rc = rho; rc(~cerca) = NaN;
        f = median(rc, 2, 'omitnan');
        f(~isfinite(f) | f <= 0) = Inf;    % sin datos -> se rechaza

        % El factor NO se deja libre, y esto es lo importante.
        %
        % Con f libre, la comprobacion solo dice "los tamanos son coherentes
        % entre si", y eso lo cumple sin esfuerzo una hipotesis totalmente
        % equivocada: si los tres crateres estan mal emparejados pero
        % comparten el mismo factor erroneo, son coherentes. Se pierde toda
        % la capacidad de discriminar. Medido: aparecian soluciones a 200-450
        % km de altitud, o sea con la escala tres veces mal, y se aceptaban.
        %
        % El inflado real esta acotado: se midio entre 1.10 y 1.45 segun el
        % giro. Exigiendo que f caiga en un rango plausible se recupera el
        % anclaje absoluto sin volver a depender del valor exacto.
        % (a) ANCLA ABSOLUTA, por deteccion. Cada crater corroborado tiene
        %     que medir lo que dice el mapa, salvo el inflado conocido.
        absOk = rho >= params.inflado_min & rho <= params.inflado_max;

        % (b) COHERENCIA RELATIVA. Ademas, los tamanos deben concordar
        %     entre si dentro de la hipotesis.
        fuera = f < params.inflado_min | f > params.inflado_max;
        relOk = abs(rho ./ f - 1) <= params.verif_diam;
        relOk(fuera, :) = false;

        tamOk = absOk & relOk;
    else
        tamOk = abs(Dmap - esperado) <= params.verif_diam * max(Dmap, 1e-9);
    end
    v = cerca & tamOk;

    % cada crater del mapa cuenta una sola vez
    neg = repmat(-(1:nd), numel(r), 1);
    idq = im; idq(~v) = neg(~v);
    idq = sort(idq, 2);
    rep = sum(idq(:,2:end) == idq(:,1:end-1) & idq(:,2:end) > 0, 2);

    corrob(r) = sum(v,2) - rep;
    VAL(r,:)  = v;
    IMIN(r,:) = im;
end

best = max(corrob);
if best < params.corrob_min, return; end

k0 = find(corrob >= best);
if numel(k0) > 1
    dd2 = abs(pos(k0) - pos(k0).');
    [~, jj] = max(sum(dd2 < params.cluster_km, 2));
    b = k0(jj);
else
    b = k0(1);
end

% ---- refinado ----
% Solo con parejas cuya caja NO toca el borde: una caja recortada tiene el
% centro desplazado hacia dentro y sesga la escala (+2.85 km medidos).
v = VAL(b,:).';
usa = v & noBorde;
if sum(usa) < 3, usa = v; end
if sum(usa) < 3, return; end

ids = IMIN(b, usa).';
src = s_det(usa);  dst = mapa.xy(ids);
ms = mean(src); md = mean(dst);
sc = src - ms;   dc = dst - md;
den = sum(abs(sc).^2);
if den <= 0, return; end
a = sum(conj(sc) .* dc) / den;
if ~isfinite(a) || abs(a) <= 0, return; end

p = a * (centro - ms) + md;

% La escala mide la distancia a los CENTROS de los crateres, que estan
% hundidos (los 258 del mapa tienen z medio de -2.35 km). La altitud sobre
% el nivel de referencia es esa distancia mas la z media de los crateres
% emparejados, que el mapa conoce. Sin esta correccion la altitud sale
% sistematicamente unos 2.7 km alta.
prof = params.frameWidth * abs(a) / (2*tand(params.FOV_horizontal/2));

R.x_km        = real(p);
R.y_km        = imag(p);
R.altura_km   = prof + mean(mapa.z(ids));
R.guinada_deg = rad2deg(angle(a));
R.corroborados = best;
R.n_hipotesis  = nh;
R.ids_mapa    = ids(:).';
R.idx_det     = find(usa).';

end
