# NWJSSP

## Guía de ejecución

La libreta nwjssp_umdac_morfin_orozco.ipynb esta lista para ser ejecutada en el entorno colab de google
* [colab](https://colab.research.google.com/)

Solo es necesario importar la libreta y ejecutar celda por celda para hacer funcionar el algoritmo

En caso de que se desee ejecutar la libreta en una maquina local es necesario contar con lo siguiente:

## Software y paquetes necesarios

- Python version 3.7 o superior
- Editor de libretas Jupyter Notebook
- Instalador de paquetes pip
- matplotlib
- pandas
- interval
- numpy
- scipy
- seaborn

> Para instalar un paquete de python con pip, regularmente se ejecuta el siguiente comando

```
pip install package_name
```
> donde package_name es reemplazado por los paquetes listados en la lista anterior (para instalar 'interval' se utiliza el package_name "pyinterval" y no "interval")

## Modificaciones adicionales

Si se desea ejecutar la libreta en una maquina local, evitar ejecutar las primeras 2 celdas de la libreta:

```python
!pip install pyinterval
```

```python
!wget https://raw.githubusercontent.com/spartan-lnx/NWJSSP/master/instances/ft06.txt
!wget https://raw.githubusercontent.com/spartan-lnx/NWJSSP/master/instances/ft10.txt
!wget https://raw.githubusercontent.com/spartan-lnx/NWJSSP/master/instances/la05.txt
!wget https://raw.githubusercontent.com/spartan-lnx/NWJSSP/master/instances/la33.txt
!wget https://raw.githubusercontent.com/spartan-lnx/NWJSSP/master/instances/la40.txt
```
Ademas, en la celda que contiene la funcion `test` para mandar a ejecutar el algoritmo

```python
ft06 = test(filename='ft06.txt',gen_size=30,pop_size=60,runs=1,tournament_size=5,bks=73)
```
editar el parametro `filename` agregando `instances/` antes del nombre de la instancia.

```python
ft06 = test(filename='instances/ft06.txt',gen_size=30,pop_size=60,runs=1,tournament_size=5,bks=73)
```