function mapa = cargar_mapa(rutaCsv, params)
% CARGAR_MAPA  Lee el mapa embarcado y construye la tabla de tripletas.
%
% Entrada: mapa_referencia*.csv -> id, nombre, tipo, x_km, y_km, z_km, D_km.
% Todo en kilometros.
%
% Poda. Una tripleta solo sirve si sus tres crateres se ven a la vez en un
% frame, asi que hay que acotar el lado maximo:
%   params.factor_lado = 0   lado maximo fijo (params.lado_max_km). Con 258
%     crateres y 200 km quedan 65.773 tripletas de 2.829.056 posibles.
%   params.factor_lado = 30  lado maximo = 30 * diametro del menor de los
%     tres, con tope en lado_max_km. Necesario en mapas densos: una tripleta
%     de crateres de tamano D solo se usa a la altura en que esos son los
%     mayores del encuadre, y ahi la huella mide del orden de 30*D. Deja
%     101.106 tripletas con 258 crateres, 362.095 con 440 y 742.446 con 753.
%
% No se usa nchoosek: con 753 crateres son 70.876.376 filas, 1,7 GB solo de
% indices. En su lugar se recorre cada crater como el menor de la tripleta y
% se buscan pares entre los mayores que le caen cerca, generando unicamente
% las que sobreviven a la poda.
%
% Descriptor: s0/s2 y s1/s2 dan la forma del triangulo, dA/dC y dB/dC los
% tamanos relativos. Los diametros son imprescindibles: con la forma sola la
% tripleta mediana tiene 97 competidoras dentro de tolerancia, y anadiendolos
% baja a 2. La forma canonica etiqueta cada vertice por el lado opuesto, de
% menor a mayor, asi que la correspondencia crater-deteccion queda resuelta al
% casar la tripleta y no hay que probar las 6 permutaciones.

if nargin < 2, params = struct(); end
if ~isfield(params, 'lado_max_km'), params.lado_max_km = 200.0; end
if ~isfield(params, 'factor_lado'), params.factor_lado = 0;     end
if ~isfield(params, 'cache'),       params.cache = '';          end

T = readtable(rutaCsv);

mapa.x  = T.x_km(:);
mapa.y  = T.y_km(:);
if ismember('z_km', T.Properties.VariableNames)
    mapa.z = T.z_km(:);
else
    warning('El mapa no trae z_km: la altitud saldra unos 2.7 km alta.');
    mapa.z = zeros(size(mapa.x));
end
mapa.D  = T.D_km(:);
mapa.n  = numel(mapa.x);
mapa.xy = complex(mapa.x, mapa.y);

fprintf('Mapa: %d crateres, D de %.1f a %.1f km\n', ...
    mapa.n, min(mapa.D), max(mapa.D));

% ---- cache ----
if ~isempty(params.cache) && isfile(params.cache)
    S = load(params.cache);
    if isfield(S,'trip') && S.trip.n_crateres == mapa.n ...
            && abs(S.trip.lado_max_km - params.lado_max_km) < 1e-9 ...
            && isfield(S.trip,'factor_lado') ...
            && abs(S.trip.factor_lado - params.factor_lado) < 1e-9
        mapa.trip = S.trip;
        fprintf('  tripletas leidas de cache: %d\n', size(mapa.trip.ids,1));
        return;
    end
end

fprintf('  construyendo tabla de tripletas...\n');
tic;

% se ordena de mayor a menor diametro: asi "los mas grandes que i" son
% simplemente los de indice menor que i
[Dord, ord] = sort(mapa.D, 'descend');
Px = mapa.x(ord);  Py = mapa.y(ord);

cap = 4e6;                       % se amplia solo si hace falta
IDS = zeros(cap, 3, 'int32');
LAD = zeros(cap, 3);
nT  = 0;

for i = 2:mapa.n                 % i es el MENOR de la tripleta
    if params.factor_lado > 0
        r = min(params.factor_lado * Dord(i), params.lado_max_km);
    else
        r = params.lado_max_km;
    end

    dx = Px(1:i-1) - Px(i);
    dy = Py(1:i-1) - Py(i);
    vec = find(dx.*dx + dy.*dy <= r*r);
    m = numel(vec);
    if m < 2, continue; end

    Qx = Px(vec);  Qy = Py(vec);
    dd = hypot(Qx - Qx.', Qy - Qy.');
    [a, b] = find(triu(dd <= r, 1));        % pares que tambien caben
    if isempty(a), continue; end

    p = vec(a);  q = vec(b);
    nuevos = numel(p);

    % lados OPUESTOS a cada vertice del triangulo (i, p, q)
    L = [ hypot(Px(p)-Px(q), Py(p)-Py(q)), ...      % opuesto a i
          hypot(Px(i)-Px(q), Py(i)-Py(q)), ...      % opuesto a p
          hypot(Px(i)-Px(p), Py(i)-Py(p)) ];        % opuesto a q

    if nT + nuevos > size(IDS,1)
        IDS = [IDS; zeros(max(nuevos, cap), 3, 'int32')]; %#ok<AGROW>
        LAD = [LAD; zeros(max(nuevos, cap), 3)];          %#ok<AGROW>
    end

    IDS(nT+1:nT+nuevos, :) = int32([repmat(i, nuevos, 1), p(:), q(:)]);
    LAD(nT+1:nT+nuevos, :) = L;
    nT = nT + nuevos;
end

IDS = IDS(1:nT, :);
LAD = LAD(1:nT, :);

% ---- orden canonico: vertices por su lado opuesto, de menor a mayor ----
[Ls, o] = sort(LAD, 2);
lin = (1:nT).';
ids = [ IDS(sub2ind(size(IDS), lin, o(:,1))), ...
        IDS(sub2ind(size(IDS), lin, o(:,2))), ...
        IDS(sub2ind(size(IDS), lin, o(:,3))) ];
ids = double(ids);

Ds = [ Dord(ids(:,1)), Dord(ids(:,2)), Dord(ids(:,3)) ];

trip.ids   = reshape(ord(ids), [], 3);   % de vuelta a indices del fichero
trip.desc  = [ Ls(:,1)./Ls(:,3), Ls(:,2)./Ls(:,3), ...
               Ds(:,1)./Ds(:,3), Ds(:,2)./Ds(:,3) ];
trip.s2_km = Ls(:,3);
trip.n_crateres  = mapa.n;
trip.lado_max_km = params.lado_max_km;
trip.factor_lado = params.factor_lado;

% se ordena por la primera componente para poder buscar por biseccion
[trip.desc(:,1), o2] = sort(trip.desc(:,1));
trip.desc(:,2:4) = trip.desc(o2, 2:4);
trip.ids   = trip.ids(o2, :);
trip.s2_km = trip.s2_km(o2);

mapa.trip = trip;
fprintf('  %d tripletas en %.1f s\n', nT, toc);

if ~isempty(params.cache)
    save(params.cache, 'trip', '-v7.3');
end

end
