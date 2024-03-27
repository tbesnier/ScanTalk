## **ScanTalk**

Official PyTorch implementation

> **ScanTalk: 3D Talking head from Unregistered Scans**.
>
> Federico Nocentini<span>&#42;</span>,  Thomas Besnier<span>&#42;</span>, Claudio Ferrari, Sylvain Arguillère, Stefano Berretti, Mohamed Daoudi
>
> <a href='https://arxiv.org/abs/2403.10942'><img src='https://img.shields.io/badge/arXiv-refs-red'></a> <a href='https://tbesnier.github.io/projects/scantalk/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<div class="row">
<p align="center">
  	<img src="https://github.com/tbesnier/ScanTalk/blob/main/gifs/scan.gif" alt="animated_scan" style="width:20%"/>
  	<img src="https://github.com/tbesnier/ScanTalk/blob/main/gifs/arnold.gif" alt="animated_arnold" style="width:20%"/>
</p>
</div>

## **Environment**

Tested with Python 3.10 on Linux.

Mandatory Python packages:
- pytorch
- trimesh
- transformers
- MPI-mesh (pip install git+https://github.com/MPI-IS/mesh.git)
- pip install pyrender
- pip install robust_laplacian potpourri3d
- pip install sacremoses

## **Demo**
- First, download the pretrained model at https://drive.google.com/file/d/1Z30PEkiPDv8Cs8xfbd35YAbN-FmeaZy6/view?usp=sharing and put it in "../pretrained_model"

- to animate a mesh in ANY topology, you first need to align it with the training meshes (use blender or ICP). 
Be aware that the upper and lower lips need to be clearly separated

- then, run:
	```
	demo.py --reference_mesh_file="..." --sample_audio="..."
	```
 
## **Training**

## **Testing**

## **Citation**
	@misc{nocentini2024scantalk,
      title={ScanTalk: 3D Talking Heads from Unregistered Scans}, 
      author={Federico Nocentini and Thomas Besnier and Claudio Ferrari and Sylvain Arguillere and Stefano Berretti and Mohamed Daoudi},
      year={2024},
      eprint={2403.10942},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
Please, cite this paper if you use it in your own work.

## **Acknowledgement**
This code highly rely on previous work such as DiffusionNet (https://github.com/nmwsharp/diffusion-net) and FaceDiffuser (https://github.com/uuembodiedsocialai/FaceDiffuser).