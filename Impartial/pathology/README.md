# ImPartial Monai Label Integration

## Pre-requisites

First you need to install `monailabel`.

Create a Python virtual environment
```shell
python3 -m venv venv
source venv/bin/activate
```
and then follow [these instructions](https://docs.monai.io/projects/label/en/latest/installation.html#from-github)
to install Monai Label from Github (weekly release).
   
## Prepare dataset
This implementation uses the `Vectra_2CH` dataset. Monai Label uses a special
folder organization for the dataset and labels, or scribbles in our case.
In order for Monai Label to pick up the scribbles as labels, you need to move
them under `Vectra_2CH/labels/final/` and rename them with original image name.

```shell
Vectr_2CH/
  image0.npz
  image1.npz
  ...
  /labels
    /final
      image1.npz
      ...
```

## Run Monai Label app

```shell
cd Impartial
monailabel start_server -a pathology -s ../Data/Vectra_2CH
```

## Test the API

There are two ways of testing out the API. You could either navigate to
`http://localhost:8000` and use the Swagger UI to interact with the API.
Or there's also a Jupyter Notebook `monai_callflow` that replicates the one
of a typical Monai Label client like the one described [here](https://docs.monai.io/projects/label/en/latest/appdeployment.html#application-call-flow).