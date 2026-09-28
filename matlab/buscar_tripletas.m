function idx = buscar_tripletas(trip, q, tolF, tolD)
% BUSCAR_TRIPLETAS  Tripletas del mapa compatibles con el descriptor
% q = [s0/s2, s1/s2, dA/dC, dB/dC].
%
% La tabla esta ordenada por la primera componente: se acota con busqueda
% binaria y solo se filtra ese trozo, en vez de recorrer las 65.773 filas en
% cada una de las 35 consultas por frame. No necesita ningun toolbox.

lo = busca_izq(trip.desc(:,1), q(1) - tolF);
hi = busca_izq(trip.desc(:,1), q(1) + tolF) - 1;

if hi < lo
    idx = zeros(0,1);
    return;
end

r = (lo:hi).';
ok =  abs(trip.desc(r,2) - q(2)) <= tolF ...
    & abs(trip.desc(r,3) - q(3)) <= tolD ...
    & abs(trip.desc(r,4) - q(4)) <= tolD;

idx = r(ok);

end


function i = busca_izq(v, x)
% Primer indice con v(i) >= x, por biseccion.
lo = 1; hi = numel(v) + 1;
while lo < hi
    mid = floor((lo + hi) / 2);
    if v(mid) < x
        lo = mid + 1;
    else
        hi = mid;
    end
end
i = lo;
end
