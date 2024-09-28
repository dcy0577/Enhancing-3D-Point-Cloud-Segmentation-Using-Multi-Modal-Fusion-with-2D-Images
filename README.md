# Enhancing-3D-Point-Cloud-Segmentation-Using-Multi-Modal-Fusion-with-2D-Images

This is implementation for the [paper](https://mediatum.ub.tum.de/doc/1691326/1691326.pdf):
> Du, C., Torres, V., Pan, Y., & Borrmann, A. (2022). MV-KPConv: Multi-view KPConv For Enhanced 3D Point Cloud Semantic Segmentation Using Multi-Modal Fusion With 2D Image. In European Conference on Product and Process Modeling 2022.

## Installation
All experiments are run on Ubuntu 16.04 with a 5GB Quadro P2000 GPU and 32 GB RAM. 
- CUDA 10.0
- cuDNN 7.6.4
- Pytorch 1.2.0
- Python 3.6
1. Create a mini conda environment provided by mvpnet to install all packages needed
```
cd <root of this repo>
conda env create --file environment.yml
conda activate mvpnet
export PYTHONPATH=$(pwd):$PYTHONPATH
```
2. Compile the module for PointNet++:
```
cd <root of this repo>
bash compile.sh
```
3. Compile the C++ extension modules for KPConv located in `KPConv-Pytorch/cpp_wrappers`.
```
cd KPConv-Pytorch/cpp_wrappers
sh compile_wrappers.sh
```
## Download ScanNet dataset
1. Run the modified download script, it will read the scene ids in the `sceneWithDoorWindow.txt` file and automatically download their 2D3D data. To save disk space, we use a custom ScanNet dataset containing a total of 146 scenes.
```
cd ScanNet
python download-scannet.py -o <root of this repo>/ScanNet --id 0
```
2. After the download, your folder should look like this:
```
<ScanNet root>/scans/<scan_id>                                     % scan id, e.g.: scene0000_00
<ScanNet root>/scans/<scan_id>/<scan_id>_vh_clean_2.ply            % 3D point cloud (whole scene)
<ScanNet root>/scans/<scan_id>/<scan_id>_vh_clean_2.labels.ply     % 3D semantic segmentation labels
<ScanNet root>/scans/<scan_id>/<scan_id>.sens                      % 2D information
<ScanNet root>/scans/<scan_id>/<scan_id>_2d-label.zip              % 2D labels
```
3. The .sens files (one per scene) need to be extracted to get the 2D frame information (color, depth, intrinsic, pose). For extraction, we need to use the official SensReader code from the ScanNet repo which is in Python 2. It is suggested to create an mini conda environment (e.g. conda create --name py27 python=2.7 ) and use it to run the convenience script provided by mvpnet using multiprocessing for extraction (please change the data paths in the script accordingly):
```
# only for this script, use Python 2.7
cd mvpnet/data/preprocess
python extract_raw_data_scannet.py
```
4. After that, you need to unzip the 2D labels. Use convenience script provided by mvpnet (adapt paths):
```
cd mvpnet/data/preprocess
python unzip_2d_labels.py
```
5. After extraction, you should have the following structure in your data directory:
```
# 2D
<ScanNet root>/scans/<scan_id>                                     % scan id, e.g.: scene0000_00 
<ScanNet root>/scans/<scan_id>/color/                              % RGB images
<ScanNet root>/scans/<scan_id>/depth/                              % depth images
<ScanNet root>/scans/<scan_id>/intrinsic/                          % camera intrinsics
<ScanNet root>/scans/<scan_id>/label/                              % 2D labels
<ScanNet root>/scans/<scan_id>/pose/                               % pose for each 2D frame
# 3D
<ScanNet root>/scans/<scan_id>/<scan_id>_vh_clean_2.ply            % 3D point cloud (whole scene)
<ScanNet root>/scans/<scan_id>/<scan_id>_vh_clean_2.labels.ply     % 3D semantic segmentation labels
```
6. In order to save disk space and reduce data loading times, we resize all images to the target resolution of 160x120. Use the following script (adapt paths in the script):
```
cd mvpnet/data/preprocess
python resize_scannet_images.py
```
## Build data cache
1. Copy the comparably lightweight 3D data, poses and intrinsics over to the directory with the resized scans with the `--parents` option in order to use the downscaled data for preprocessing
```
cd <ScanNet root>/scans
cp --parents scene0*/*.ply <ScanNet root>/scans_resize_160x120/
cp -r --parents scene0*/intrinsic <ScanNet root>/scans_resize_160x120/
cp -r --parents scene0*/pose <ScanNet root>/scans_resize_160x120/
```
2. if you only want to run MV-KPConv, use following script to dump all the ply files (3D point clouds) into one pickle file for faster loading (adapt paths in the script)
```
python mvpnet/data/preprocess/preprocess.py -o <root of this repo>/ScanNet/cache_rgbd -s train
python mvpnet/data/preprocess/preprocess.py -o <root of this repo>/ScanNet/cache_rgbd -s val
```
   if you also want to run mvpnet, you need to compute the overlap of each RGBD frame with the whole scene point cloud here, add `--rgbd` whenn run the script
```
python mvpnet/data/preprocess/preprocess.py -o <root of this repo>/ScanNet/cache_rgbd -s train --rgbd
python mvpnet/data/preprocess/preprocess.py -o <root of this repo>/ScanNet/cache_rgbd -s val --rgbd
```
## Training
### 2D model
Train the 2D networks on the 2D semantic segmentation task, adpat path in `unet_resnet34.yaml`
```
python mvpnet/train_2d.py --cfg configs/scannet/unet_resnet34.yaml
```
You can also use pretrained 2D model, simply download it and save it in `outputs/scannet/unet_resnet34/`
### MV-KPConv
Adapt path in `KPConv-PyTorch/utils/config.py`

https://github.com/dcy0577/Enhancing-3D-Point-Cloud-Segmentation-Using-Multi-Modal-Fusion-with-2D-Images/blob/5d0ac3d35553429f7a388dbc1fa1b681bda3db41/KPConv-PyTorch/utils/config.py#L42-L51

Train the early fusion version:
```
cd KPConv-PyTorch
python train_ScanNet_sphere.py
```
Train the middle fusion version:
```
cd KPConv-PyTorch
python train_ScanNet_sphere_middle_fusion.py
```
Train the late fusion version:
```
cd KPConv-PyTorch
python train_ScanNet_sphere_late_fusion.py
```
When you start a new training, it is saved in `KPConv-PyTorch/results` folder. A dated log folder will be created, containing many information including loss values, validation metrics, model checkpoints, etc. You can use `plot_convergence.py` to plot a logged training, please adapt the path and import dataset
```
cd KPConv-PyTorch
python plot_convergence.py.py
```
### KPConv baseline
Adapt path in `KPConv-PyTorch/utils/config.py`
Train the KPConv baseline model:
```
cd KPConv-PyTorch
python train_ScanNet_baseline.py
```
### mvpnet baseline
Adpat path in `mvpnet_3d_unet_resnet34_pn2ssg.yaml`
Train mvpnet baseline model
```
python mvpnet/train_mvpnet_3d.py --cfg configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml
```
## Testing
### Test MV-KPConv
Three versions can all be tested in `test_models.py`. You will find detailed comments explaining how to choose which logged trained model you want to test. For different versions, these lines may need to be adjusted:

https://github.com/dcy0577/Enhancing-3D-Point-Cloud-Segmentation-Using-Multi-Modal-Fusion-with-2D-Images/blob/5d0ac3d35553429f7a388dbc1fa1b681bda3db41/KPConv-PyTorch/test_models.py#L43-L45

https://github.com/dcy0577/Enhancing-3D-Point-Cloud-Segmentation-Using-Multi-Modal-Fusion-with-2D-Images/blob/5d0ac3d35553429f7a388dbc1fa1b681bda3db41/KPConv-PyTorch/test_models.py#L157-L160

```
cd KPConv-PyTorch
python test_models.py
```
### Test KPConv baseline
```
cd KPConv-PyTorch
python test_scannet_baseline_model.py
```
### Test mvpnet baseline
```
python mvpnet/test_mvpnet_3d.py --cfg configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml  --num-views 5
```
## Results
|  Network   | mIoU  |
|  ----  | ----  |
| MV-KPConv  | 74.40 |
| KPConv baseline | 52.58 |
| MVPNet baseline | 71.21 |

Note that results can slightly differ from training to training, the testing result of KPConv baseline & MV-KPConv may also fluctuate slightly because we use a voting mechanism.
## Pretrained model
The pretrained model can be download [here](https://syncandshare.lrz.de/getlink/fiNzkbd6y7YLLoqxAftGvA6K/).
pretrained 2D model should be saved in `outputs/scannet/unet_resnet34/`

pretrained mvpnet should be saved in `outputs/scannet/mvpnet_3d_unet_resnet34_pn2ssg/`.

pretrained MV-KPConv should be saved in `KPConv-PyTorch/results`.

## Update 4/2022: Add dataloader for custom dataset (Colmap)
We used a laser scanner to acquire point clouds of interior rooms and a camera to take multi-view color pictures of the rooms. We used colmap to perform 3D reconstruction using these images to predict the depth map and camera parameters corresponding to the images. Using this 2D-3D information we constructed a custom dataset.
The data should be placed in the colmap folder with following data structure:
```
-scene(pointcloud) name
   - depth_maps                   % .geometric.bin files from colmap
   - parameter                    % images.bin, cameras.bin, points3D.bin from colmap; matrix_for_images.txt stores the correction matrix for photos in oder to align the reconstruction point cloud with the laser scanned point cloud
   - images                       % color images
   - pointcloud                   % laser scanned point cloud
```
In addition, using `KPConv-PyTorch/test_colmap_baseline_models.py` you can test the performance of KPConv on this custom datasets. Using `KPConv-PyTorch/test_models_colmap.py` you can test the performance of MV-KPConv on this custom datasets. Note: Here we used the best performing MV-KPConv and KPConv models trained on scannet for our experiments.

## Citation
```
@inproceedings{Du:2022:MV-KPConv,
	author = {Du, C. and Vega Torres, M.A. and Pan, Y. and Borrmann, A.},
	title = {MV-KPConv: Multi-view KPConv For Enhanced 3D Point Cloud Semantic Segmentation Using Multi-Modal Fusion With 2D Image},
	booktitle = {European Conference on Product and Process Modeling 2022},
	year = {2022},
	month = {Sep},
	url = {https://mediatum.ub.tum.de/doc/1691326/1691326.pdf},
}
```

## Acknowledgements
Note that the code is borrowed from [MVPNet](https://github.com/maxjaritz/mvpnet) and [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch)
