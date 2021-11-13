Wave-U-Net

#Installation steps on the cluster

* follow step 1 & 2 listed here - https://isis.tu-berlin.de/pluginfile.php/2110087/mod_resource/content/16/python_cluster_tutorial.pdf

* As we do not have root privileges on the cluster, we need a virtual environment to install python packages.
To create a virtual environment named virt_pml, type:
```
virtualenv -p python3 virt_pml
```

In this virtual environment, we can install all python packages we want to install. First, we have to activate
the virtual environment. Type:
```
source virt_pml/bin/activate
```

Then we have to install all the packages.
```
python -m pip install -r requirements.txt
```
We have to do this only once. Now we can always activate the virtual environment

* Download the dataset using the bash script by running the following command. We only have to do this once.
WARNING: THIS IS A 5GB DATASET.
```
cd data; source get_dataset.sh
```

#Usage
* todo
