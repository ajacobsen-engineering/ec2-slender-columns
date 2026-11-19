# Slanke betongsøyler etter Eurokode 2

Kontroll av slanke, armerte betongsøyler etter NS-EN 1992-1-1, bygget rundt den
generelle metoden i punkt 5.8.6.

Søylen deles i snitt langs lengden. I hvert snitt hentes krumningen fra en
ikke-lineær tverrsnittsanalyse (via [structuralcodes](https://github.com/fib-international/structuralcodes)),
og krumningen integreres til utbøyning. Det andreordens momentet oppdateres til
løsningen konvergerer — standard når endringen per iterasjon kommer under 1 %.
Kryp legges inn ved å skalere betongtøyningene med (1 + φ_ef).

Som kontroll regnes de to forenklede metodene (5.8.7 nominell stivhet og 5.8.8
nominell krumning) ved siden av, og et kapasitetssøk finner lastfaktoren der
søylen svikter.

## Installasjon

```bash
pip install structuralcodes numpy matplotlib
```

`structuralcodes` må være 0.7 eller nyere (bruker `BeamSection`).

## Bruk

```python
from slanke_soyler import Soyle, rapport

s = Soyle(
    b=400, h=400, l0=6000, fck=35,
    n_bars_per_side=4, bar_dia=25,
    N_Ed=2500e3, M0y_top=80e6, M0y_bot=40e6,
    phi_ef=1.0,
)
rapport(s, lag_plott=True)
```

Enheter er N, mm og MPa (moment i Nmm). Trykk-kraft angis positivt.

Utskriften viser iterasjonene og en kontrolltabell:

```
                              5.8.6       5.8.7       5.8.8
 e2 [mm]                       14.2           -        43.6
 M_Ed [kNm]                   127.7       158.9       203.7
 M_Rd [kNm]                   385.8       385.8       385.8
 Utnyttelse                    0.33        0.41        0.53
 Status                          OK          OK          OK
```

`lag_plott=True` lagrer M(x), v(x), κ(x) og konvergenshistorikken som PNG.

## Hva koden tar

- Rektangulært og sirkulært tverrsnitt
- Leddlagret søyle og utkrager
- Enakset og toakset bøying (5.8.9)
- Endemoment og tverrlast (jevnt fordelt og punktlast)
- Kryptall manuelt, eller beregnet etter Tillegg B
- Kapasitetssøk og batch-kjøring av flere søyler

Funksjonene `generell_metode`, `kapasitetsfaktor` og `batch` kan kalles direkte
hvis du vil bruke resultatene videre i stedet for utskriften.

## Forbehold

Verktøyet er under utvikling og bør verifiseres mot håndberegning før det brukes
i prosjektering. To ting å være klar over:

- Utkrager bruker en enklere dobbeltintegrasjon fra innspenningen enn den
  leddlagrede, som løses med et tridiagonalt ligningssystem.
- Effektiv dybde for sirkulært tverrsnitt i de forenklede metodene (5.8.7/5.8.8)
  er en tilnærming. Den generelle metoden er uansett den nøyaktige for sirkulære
  tverrsnitt.

Bruddtypen merkes «materialbrudd» når tverrsnittet ikke tar opp førsteordens
last, og «instabilitet» når utbøyningen vokser til brudd. Begge gir IKKE OK.
