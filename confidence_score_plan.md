Weil es schwer ist eine Kantenerkennung zu haben die unter allen Umständen stabil läuft wollen wir einen confidence score einführen der die Qualität der aktuellen Erkennung wiederspiegelt.
Als initiale Idee gilt folgede Rechnung für jede TBK: 
`confidence_score = 1 - (Anzahl invalider Pixel im ROI)/(Anzahl aller Pixel im ROI)`
Somit ergibt sich ein %-Score mit 100 % : alle Pixel sind valide -> detektierte Kante ist ziemlich sicher. 50 % : Die Hälfte aller Pixel im ROI sind invalide -> die detektierte Kante ist potentiell durch invalide Bereiche (z.B. wegen starker Sonneneinstrahlung) entstanden, somit ist diese Kante nicht "sicher".

**Problem:** Die Bereiche die durch Sonneneinstrahlung die Tiefenerkennung beeinträchtigen liefern nicht immer invalide Werte, sondern oft einfach falsche Werte.


Als alternative Berechnung für den Confidence lässt sich folgendes implementieren: Vergleiche die detektierte Kante über mehrere Frames hinweg. Falls sich die Kante über mehrere Frames hinweg stark verändert deutet dies auf eine schwache Genauigkeit hin. Sind die Koordinaten der aktuell detektierten Linie hingegen ähnlich zu den Linien in den vorherigen Frames deutet dies auf eine stabile Detektion hin.
