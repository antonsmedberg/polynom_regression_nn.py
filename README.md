# Polynom Regression med Neuralt Nätverk

## Introduktion
Detta projekt implementerar ett neuralt nätverk för regression baserat på TensorFlow för att analysera och lära sig från syntetiska data genererade från en polynomfunktion. Genom att följa processen från datagenerering till modellträning och utvärdering, demonstrerar skriptet kraften i neurala nätverk för att förutsäga och tolka komplexa datamönster, stödd av visualiseringar för en djupare insikt.

## Innehåll
- [Förutsättningar](#förutsättningar)
- [Installation](#installation)
- [Användning](#användning)
- [Konfiguration](#konfiguration)
- [Exempel och Situationer](#exempel-och-situationer)
    - [Experiment med olika polynomfunktioner](#experiment-med-olika-polynomfunktioner)
    - [Användning av Modellen för Prediktioner](#användning-av-modellen-för-prediktioner)
- [Integration](#integration)
- [Bidrag](#bidrag)
- [Licens](#licens)

## Förutsättningar
För att framgångsrikt köra detta projekt behöver du:
- Python 3.6 eller senare
- TensorFlow 2.0 eller senare
- NumPy
- Matplotlib

## Installation
Installera nödvändiga bibliotek med pip:
```bash
pip install numpy tensorflow matplotlib
```

## Användning
Kör skriptet genom att navigera till dess katalog och exekvera:
```bash
python polynom_regression_nn.py
```

## Konfiguration
Anpassa neurala nätverkets och datagenereringens parametrar genom `CONFIG`-dictionaryn, med möjlighet att justera bland annat:
- `neurons_layer_1`, `neurons_layer_2`: Antalet neuroner i de dolda lagren.
- `epochs`, `batch_size`, `learning_rate`, `dropout_rate`: Träningsparametrar.
- `x_start`, `x_end`, `num_points`: Dataomfång och mängd.

## Exempel och Situationer
### Experiment med olika polynomfunktioner
Utforska nätverkets anpassningsförmåga genom att modifiera `generate_data`-funktionens polynomfunktion.

### Användning av Modellen för Prediktioner
Använd den tränade modellen för prediktioner på ny data genom TensorFlow och `.predict`-metoden.

## Integration
Inkludera detta neuralt nätverk i ditt projekt genom att lägga till skriptet i kodbasen och installera de nödvändiga beroendena.

## Licens
Anpassa denna sektion med ditt projekts licensinformation.
