// Plain-language captions so the visualisations actually teach. `does` = what
// the layer does in one breath; `read` = how to read the picture below it.
// Keyed by layer type, with a category-level fallback.

export type Explain = { does: string; read: string }

export const BY_TYPE: Record<string, Explain> = {
  Input: {
    does: 'Hier kommen die Rohdaten ins Modell — die Zahlen, mit denen alles anfängt.',
    read: 'Jeder Balken (oder jede Kachel) ist ein Eingabewert. Höhe/Helligkeit = wie groß die Zahl ist.',
  },
  Output: {
    does: 'Das Endergebnis des Modells nach allen Layern.',
    read: 'Diese Werte sind die finale Antwort — z. B. ein Score pro Klasse.',
  },
  Linear: {
    does: 'Jedes Ausgangs-Neuron ist eine gewichtete Mischung ALLER Eingänge (multiplizieren + aufsummieren).',
    read: 'Links die Eingangs-Neuronen, rechts die Ausgänge. Helligkeit eines Punkts = wie stark er gerade „feuert". '
      + 'Jede Linie ist ein gelerntes Gewicht: rot zieht den Ausgang hoch (+), blau zieht runter (−), dicker = stärker. '
      + 'Die Heatmap darunter ist die komplette Gewichtsmatrix.',
  },
  ReLU: {
    does: 'Lässt positive Werte unverändert durch und setzt alle negativen auf 0 — die einfachste „Nichtlinearität".',
    read: 'Die Kurve ist die Funktion. Jeder Punkt ist ein echter Wert aus dem Layer davor: links von 0 fällt er auf 0 '
      + '(„abgeschnitten"), rechts bleibt er. Die Sparsity-Angabe sagt, wie viel Prozent auf 0 gesetzt wurden.',
  },
  GELU: {
    does: 'Weiche Variante von ReLU: drückt negative Werte sanft Richtung 0, statt hart abzuschneiden.',
    read: 'Die Kurve ist die Funktion, die Punkte sind echte Eingabewerte. Beachte den weichen Knick um 0.',
  },
  SiLU: {
    does: 'Weiche, geschwungene Variante von ReLU (auch „Swish").',
    read: 'Die Kurve ist die Funktion, die Punkte sind echte Eingabewerte.',
  },
  Sigmoid: {
    does: 'Quetscht jede Zahl in den Bereich 0…1 — praktisch für Wahrscheinlichkeiten.',
    read: 'Große negative Werte landen nahe 0, große positive nahe 1. Die Punkte zeigen, wo die echten Werte landen.',
  },
  Tanh: {
    does: 'Quetscht jede Zahl in den Bereich −1…1 (wie Sigmoid, nur symmetrisch um 0).',
    read: 'Die Punkte zeigen, wo die echten Eingabewerte auf der S-Kurve landen.',
  },
  Softmax: {
    does: 'Macht aus den Roh-Scores eine Wahrscheinlichkeitsverteilung, die sich zu 1 summiert.',
    read: 'Jeder Balken ist die Wahrscheinlichkeit für eine Klasse. Der höchste Balken ist die Vorhersage des Modells.',
  },
  LogSoftmax: {
    does: 'Wie Softmax, gibt aber den Logarithmus der Wahrscheinlichkeiten zurück (numerisch stabiler fürs Training).',
    read: 'Werte sind ≤ 0; der größte (am nächsten an 0) ist die wahrscheinlichste Klasse.',
  },
  Conv2d: {
    does: 'Schiebt kleine Filter übers Bild und erkennt lokale Muster — Kanten, Ecken, Texturen.',
    read: 'Jede Kachel oben ist das Ergebnis EINES Filters (eine „Feature-Map"): helle Stellen = dort wurde das Muster '
      + 'gefunden. Unten die gelernten Filter selbst (rot/blau = Vorzeichen der Gewichte).',
  },
  Flatten: {
    does: 'Sortiert die Zahlen nur um (z. B. ein 2D-Bild → eine lange 1D-Liste). Es wird nichts gerechnet.',
    read: 'Dieselben Werte wie davor, nur in einer Reihe.',
  },
  Embedding: {
    does: 'Schlägt für jede Token-ID einen gelernten Vektor nach — verwandelt Wörter/Symbole in Zahlen.',
    read: 'Jeder Balken ist eine Dimension des nachgeschlagenen Vektors.',
  },
  MaxPool2d: {
    does: 'Behält in jedem kleinen Fenster nur den größten Wert — verkleinert das Bild, hält die stärksten Signale.',
    read: 'Links die Feature-Maps vorher, rechts danach: gleiche Muster, aber kleiner/gröber.',
  },
  AvgPool2d: {
    does: 'Mittelt jedes kleine Fenster zu einem Wert — verkleinert das Bild, glättet es.',
    read: 'Links vorher, rechts danach: gleiche Muster, kleiner und weicher.',
  },
  AdaptiveAvgPool2d: {
    does: 'Mittelt jede Feature-Map auf eine feste Zielgröße herunter (oft 1×1 = ein Wert pro Kanal).',
    read: 'Links die Maps vorher, rechts stark verkleinert.',
  },
  BatchNorm2d: {
    does: 'Zentriert + skaliert jede Kanal-Verteilung (Mittel ≈ 0, Streuung ≈ 1) — stabilisiert das Training.',
    read: 'Histogramm vorher vs. nachher: die Werte werden um 0 zentriert und gleichmäßiger.',
  },
  BatchNorm1d: {
    does: 'Zentriert + skaliert jede Feature-Verteilung (Mittel ≈ 0, Streuung ≈ 1).',
    read: 'Histogramm vorher vs. nachher: zentriert und gleichmäßiger.',
  },
  LayerNorm: {
    does: 'Normalisiert jede einzelne Probe über ihre Features (Mittel ≈ 0, Streuung ≈ 1).',
    read: 'Histogramm vorher vs. nachher: die Verteilung wird um 0 zentriert.',
  },
  Add: {
    does: 'Addiert mehrere Eingänge elementweise (z. B. Residual-/Skip-Verbindung).',
    read: 'Die Eingänge oben, ihre Summe unten.',
  },
  Concat: {
    does: 'Hängt mehrere Eingänge aneinander zu einem längeren/breiteren Tensor.',
    read: 'Die Eingänge oben werden hintereinandergehängt → unten das Ergebnis.',
  },
  Multiply: {
    does: 'Multipliziert mehrere Eingänge elementweise (Gating/Maskierung).',
    read: 'Die Eingänge oben, ihr elementweises Produkt unten.',
  },
}

export const BY_CATEGORY: Record<string, Explain> = {
  Conv: {
    does: 'Schiebt gelernte Filter über die Eingabe und erkennt lokale Muster.',
    read: 'Jede Kachel = Ergebnis eines Filters; helle Stellen = Muster gefunden. Unten die Filter selbst.',
  },
  Linear: {
    does: 'Mischt die Eingänge gewichtet zu neuen Werten.',
    read: 'Höhe/Helligkeit = Stärke des Werts.',
  },
  Activation: {
    does: 'Eine Nichtlinearität: biegt die Werte, damit das Netz mehr als nur Geraden lernen kann.',
    read: 'Die Kurve ist die Funktion, die Punkte sind echte Werte aus dem Layer davor.',
  },
  Norm: {
    does: 'Zentriert und skaliert die Werte, damit das Training stabil und schnell bleibt.',
    read: 'Höhe/Helligkeit = Stärke; die Verteilung wird gleichmäßiger als davor.',
  },
  Pool: {
    does: 'Fasst benachbarte Werte zusammen (Maximum oder Mittel) und verkleinert das Bild.',
    read: 'Wie die Feature-Maps davor, nur gröber/kleiner.',
  },
  Reshape: {
    does: 'Ordnet die Zahlen nur neu an — es wird nichts gerechnet.',
    read: 'Dieselben Werte, andere Form.',
  },
  Merge: {
    does: 'Kombiniert mehrere Eingänge (addieren, multiplizieren oder aneinanderhängen).',
    read: 'Das Ergebnis der Zusammenführung.',
  },
  Attention: {
    does: 'Lässt jede Position auf relevante andere Positionen „schauen" und mischt deren Werte.',
    read: 'Höhe/Helligkeit = Stärke des Werts.',
  },
  Recurrent: {
    does: 'Liest die Sequenz Schritt für Schritt und trägt einen „Erinnerungs"-Zustand mit (LSTM/GRU/RNN).',
    read: 'Heatmap: Zeilen = Zeitschritte, Spalten = Hidden-Dimensionen; helle Stellen = aktive Einheiten.',
  },
  Graph: {
    does: 'Mischt für jeden Knoten die Merkmale seiner Nachbarn ein (Message Passing im Graphen).',
    read: 'Höhe/Helligkeit = Stärke des Knoten-Merkmals nach der Nachbarschafts-Aggregation.',
  },
  IO: {
    does: 'Daten-Ein- bzw. -Ausgang des Modells.',
    read: 'Höhe/Helligkeit = wie groß die Zahl ist.',
  },
}

export function explainFor(layerType: string, category: string): Explain {
  return BY_TYPE[layerType] ?? BY_CATEGORY[category] ?? {
    does: 'Eine Rechenoperation im Modell.',
    read: 'Höhe/Helligkeit der Werte zeigt ihre Stärke.',
  }
}
