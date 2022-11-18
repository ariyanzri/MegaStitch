# MegaStitch: Robust Large Scale Image Stitching

MegaStitch is a novel method for stitchig high resolution images in large scale while being robust to drift and inconsistencies. This method was originally developed as an open source tool for the Phytooracle team to geo-correct and stitch their high resolution large scale imagesets. You can find the paper [here](https://www.researchgate.net/profile/Ariyan-Zarei/publication/354153722_MegaStitch_Robust_Large_Scale_Image_Stitching/links/612803a70360302a005f3d62/MegaStitch-Robust-Large-Scale-Image-Stitching.pdf). 

This repository contains all the necessary code to run geo-correction and stitching procedures of MegaStitch method and to reproduce our results. If you find this method and the paper interesting and useful for your research, please cite us using the following bibliography. 

```
@article{zarei2021megastitch,
  title={MegaStitch: Robust Large Scale Image Stitching},
  author={Zarei, Ariyan and Gonzalez, Emmanuel and Merchant, Nirav and Pauli, Duke and Lyons, Eric and Barnard, Kobus},
  year={2021},
  publisher={TechRxiv}
}

```

This README contains instructions on how to get the data that were used in the paper, install dependencies, and run MegaStitch. 

## Data
<hr>

You can also find all the datasets as well as the Ground Control Points (GCPs) for each of them at [this](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/MegaStitch/megastitch_data.tar) link. 

## Requirements and Installation
<hr>

Currently, there are no installation scripts for this repo. In order to use MegaStitch, you need to make sure that you have all the required python packages and files, and based on your needs, you need to run one of the main entry points of the repo. You can find the list of packages we installed on our conda environment at [this text file](https://github.com/ariyanzri/MegaStitch/blob/main/requirements.txt). It is very important to install the same versions of some of these packages in order for the code to run.  

## Running MegaStitch and MGRAPH
<hr>

MegaStitch can be used to stitch and geo-correct a different variety of images. These images can either contain approximate geo-referencing, or not. We tested MegaStitch on different drone datasets and a set of specific super high resolution images of plants acquized by a moving gantry machine over an agriculture field. 

As one can read in the paper, MegaStitch have different varieties and can be run with different configurations. The main entry point script to be used for stitching and geo-correction of drone images with GPS information as well as other images captured with handheld cameras is `MegaStitch_Main.py`. This python script requires the following arguments:

* `-d / --data`: The path to the data directory. This directory should contain all the images. 

* `-r / --result`: The path to the directory where the results will be saved. Different files and folders will be created in this folder.

* `-g / --gcp`: The path to Ground Control Points (GCPs) json file. This file should have the same format as [this sample file](https://github.com/ariyanzri/MegaStitch/blob/main/GCP_sample.json).

* `-s / --settings`: The path to the json file that contains the configuration/settings information. You can find a sample of this file [here](https://github.com/ariyanzri/MegaStitch/blob/main/sample_settings.json).

In order to run the MGRAPH method, you should use `MGRAPH_Main.py` with the same arguments. 