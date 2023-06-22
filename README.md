# face-recognition

## Članovi tima:
Novica Nikolić, SW93-2016

## Definicija problema:
Prepoznavanje i identifikovanje osoba na slici/snimku.

## Motivacija problema:
Danas sve vise modernih bezbednosnih sitema koriste neki vid identifikovanja osoba radi kontrole pristupa zaposlenim u kompaniji. Rešenje problema se može primeniti u bankovnim bezbednosnim sistemima kao jedan vidi kontrole zaposlenih.
A detekcija osoba na slici je postao masovni problem za većinu društvenih mreža.
Na osnovu datih primjera primjene detekcije i identifikacije osoba, vidi se da sistem u današnjem vremenu nalazi primjenu na raznim problemima.

## Skup podatak:
Skup podataka za identifikovanje osoba ću ručno generisati na osnovu primjera.

## Metodologija:
Za realizaciju projekta koristiće se Python programski jezik i njegova OpenCV biblioteka. Pomocu CascadeClassifier i skupa tacaka koje se koriste za detekciju lica link , izvršice se detekcija lica osobe.

Na osnovu skupa podataka koji je predhodno generisan, obučiće se neuronska mreža pomću cv2.face.traing mehanizma kako bi dobili predikciju osoba na slici/snimku

## Model evaluacije:
Za model evaluacije prepoznavanja i identifikovanja osoba koristio bi: precision, recall i f1-score.
