Wave-U-Net

#Cluster account setup

* Follow step 1 listed here - https://isis.tu-berlin.de/pluginfile.php/2110087/mod_resource/content/16/python_cluster_tutorial.pdf

#Environment setup (local and cluster)
We use [miniconda](https://docs.conda.io/en/latest/miniconda.html) as package manager. To install miniconda:
1. Download the installation script for Python 3.8 and your OS (download [this](https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh) for the cluster).
2. Run the installer: `bash <script_name>` and follow instructions on screen. When asked to run `conda init`, answer `yes` if you want conda 
to be initialized at login. That will add conda initialization in your `.bashrc`. 
3. To ensure conda is initialized on ssh login to the cluster, you can call `.bashrc` from your `.bash_profile` like this: 
```
if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi
```
4. The project environment is called `pml` and its dependencies are listed in `environment.yml`. Install all environment dependencies: `conda env create -f environment.yml`


We have to do this only once. Now we can always activate the conda environment with `conda activate pml`
and deactivate it with `conda deactivate`.

# MUSDB dataset download
* Download the dataset using the bash script by running the following command. We only have to do this once.
WARNING: THIS IS A 5GB DATASET.
```
cd data; source get_dataset.sh
```

#Usage
* todo
