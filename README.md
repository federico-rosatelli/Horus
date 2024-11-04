<h1>Enviroment Monitoring by UAVs</h1>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

[Università La Sapienza Roma](https://www.uniroma1.it/), [Dipartimento di Informatica](https://www.studiareinformatica.uniroma1.it/)

![Sapienza Università di Roma](https://upload.wikimedia.org/wikipedia/commons/0/0d/Uniroma1.svg)

# SALIENCY DETECTION
## SPATIOTEMPORAL KNOWLEDGE DISTILLATION


Il progetto è suddiviso in 3 aree:

- SpatioTemporal Estimation of Aereal Video Saliency
  - Dynamic saliency estimation approach for aerial videos via spatiotemporal knowledge distillation
    - 2 teacher model
    - 2 student model
    - 1 spatiotemporal model
  - The knowledge of spatial and temporal saliency is first separately transferred from the two complex and redundant teachers to their simple and compact students.
  - The desired spatiotemporal model is further trained by distilling and encoding the spatial and temporal saliency knowledge of two students into a unified network

### TODO

# USAGE

## Build horus

Esempio di come costruire la rete neurale horus con i parametri inseriti nel file di configurazione `config/conf.yaml`

Impossibile eseguire senza dataset!

```shell
python3 horus.py build --verbose staging
```

## Run horus

Prende in input un file .avi nella cartella `tester/videos/` e restituisce come output un video .mp4 nella stessa cartella con la heatmap generata dal modello studente.

```shell
python3 horus.py run
```





