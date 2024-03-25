## **ScanTalk**

Official PyTorch implementation

> **ScanTalk: 3D Talking head from Unregistered Scans**.
>
> Federico Nocentini<span>&#42;</span>,  Thomas Besnier<span>&#42;</span>, Claudio Ferrari, Sylvain Arguillère, Stefano Berretti, Mohamed Daoudi
>
> <a ><img src='https://img.shields.io/badge/arXiv-refs-red'></a> <a href='https://tbesnier.github.io/projects/scantalk/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<video width="320" height="240" controls>
      <source src="https://github.com/tbesnier/ScanTalk/data/videos/demo.mp4" type=video/mp4>
</video>

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

## **Acknowledgement**
This code highly rely on previous work such as DiffusionNet (https://github.com/nmwsharp/diffusion-net) and FaceDiffuser (https://github.com/uuembodiedsocialai/FaceDiffuser).