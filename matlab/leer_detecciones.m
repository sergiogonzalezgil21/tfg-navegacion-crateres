function D = leer_detecciones(rutaCsv, params)
% LEER_DETECCIONES  Carga detecciones.csv y lo indexa por frame.
%
% El fichero viene de 08_exportar_detecciones.py y va en pixeles, sin ninguna
% magnitud en kilometros y sin altitud. Eso es deliberado: la altitud es lo
% que este sistema tiene que estimar, y darsela hecha invalidaria la
% validacion.
%
% Columnas: frame, cx_px, cy_px, w_px, h_px, cx_norm, cy_norm, w_norm, h_norm,
% conf.
%
% Las cajas cortadas por el borde del frame se marcan, pero no se descartan.
% Tienen el centro desplazado hacia dentro, lo que comprime la nube de puntos
% y sesga la altitud en +2.85 km, asi que se excluyen del ajuste final de
% escala; su posicion sigue siendo aproximadamente correcta y hacen falta para
% llegar a las tripletas.

if nargin < 2, params = struct(); end
if ~isfield(params, 'frameWidth'),  params.frameWidth  = 1920; end
if ~isfield(params, 'frameHeight'), params.frameHeight = 1080; end

W = params.frameWidth;
H = params.frameHeight;

T = readtable(rutaCsv);

f  = T.frame(:);
cx = T.cx_px(:);   cy = T.cy_px(:);
w  = T.w_px(:);    h  = T.h_px(:);

borde = (cx - w/2 < 1) | (cx + w/2 > W-1) | ...
        (cy - h/2 < 1) | (cy + h/2 > H-1);

fmin = min(f); fmax = max(f);
D.frame_min = fmin;
D.frame_max = fmax;
D.datos = cell(fmax - fmin + 1, 1);

for k = 1:numel(f)
    i = f(k) - fmin + 1;
    D.datos{i}(end+1, :) = [cx(k), cy(k), w(k), double(borde(k))];
end

nb = cellfun(@(c) size(c,1), D.datos);
fprintf(['Detecciones: %d cajas en %d frames (media %.1f por frame), ' ...
         '%.1f%% tocan el borde\n'], ...
    numel(f), sum(nb>0), mean(nb(nb>0)), 100*mean(borde));

end
