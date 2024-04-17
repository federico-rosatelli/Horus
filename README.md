<h1>Enviroment Monitoring by UAVs</h1>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

[Università La Sapienza Roma](https://www.uniroma1.it/en), [Dipartimento di Informatica](https://www.studiareinformatica.uniroma1.it/)

![Sapienza Università di Roma](https://upload.wikimedia.org/wikipedia/it/thumb/0/0d/Uniroma1.svg/1200px-Uniroma1.svg.png)

# Project Structure

Il progetto è suddiviso in 3 aree:

- Curiosity heatmap
  - modello ml che permetta di stabilire cosa è o non è curioso
  - dal modello del paper
- Known Object
  - creare/utilizzare ml che identifichi cose che già si conoscono (es. persone, macchine etc..)
  - quel che gia si conosce non è soggetto a uno studio successivo
- Uknown Object
  - classificare e scegliere gli oggetti che non si conoscono nel mondo
  - algoritmi euristici

## Curiosity HeatMap
### #TODO

## Known Object
L'implementazione potrebbe essere fatta tramite VisDrone.
Su kaggle esistono progetti facilmente usabili che fanno proprio al caso di questo progetto
### #TODO

## Uknownk Object

Si è realizzato un algoritmo euristico per la classificazione e la scelta completamente indipendente di ciò che potrebbe essere di interesse per una persona, così come per una macchina.

Il modello è abbastanza semplice.

Una volta preso in considerazione un oggetto nel mondo, che la Curiosity HeatMap ha identificato come potenzialmente interessante, se ne ricava la forma.

La forma viene presa tramite [rembg](https://github.com/danielgatis/rembg), un programma di AI che permette il background erasing. Estrapolata la figura da noi presa in considerazione altri algoritmi ne stabiliranno le proprietà di nostro interesse. In particolare ci interesserà sapere:
- La forma geometrica
- Il colore medio
- La dimensione
- Quando è frastagliato
- La simmetria

Una volta presi in considerazione questi parametri, ognuno con il proprio bias, il programma andrà a scegliere il moglior prossimo candidato. (#TODO la funzione di scelta ancora non è perfetta va implementata meglio)




