# units-hpc-final

Andrea Esposito (mat. SM3600005)

# Exercise 1 (22/02/2024)
Exercise 1 (MPI OSU benchmarks) resides in folder `exercise1`. Job files (sh scripts) that were used on the ORFEO cluster to collect data have been gathered in the `jobs` folder. Analysis of the four collective algorithms (Broadcast Chain/Binary Tree and Barrier Linear/Double Ring) is carried out in the respective Jupyter notebook files. The creation of a Python environment is recommended and all dependencies can be installed with

```bash
(venv) pip install -r requirements.txt
```

Results are stored in the `results` folder (exported graphs and final report in PDF format).

# Exercise 2 (23/05/2024)
Exercise 2c (hybrid Mandelbrot) resides in folder `exercise2`. Job files (sh scripts) that were used on the ORFEO cluster to collect data are stored in subfolders of the `epyc` and `thin` folders, according to which type of data (scaling) they are meant to collect. Data gathered on Epyc nodes was chosen to carry out the scalability analysis. Such analysis is carried out in a Jupyter notebook in the `results/epyc` folder. The creation of a Python environment is recommended and all dependencies can be installed with

```bash
(venv) pip install -r requirements.txt
```

Results are stored in the `results` folder (including exported graphs and final report in PDF format).