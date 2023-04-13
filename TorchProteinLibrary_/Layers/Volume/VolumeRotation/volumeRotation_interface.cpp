#include "volumeRotation_interface.h"
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/Volume/RotateGrid.h>
#include <iostream>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Math/nUtil.h>


void VolumeGenGrid( torch::Tensor rotations, torch::Tensor grid){
    CHECK_GPU_INPUT_TYPE(rotations, torch::kFloat32);
    CHECK_GPU_INPUT_TYPE(grid, torch::kFloat32);
    if(rotations.ndimension()!=3 || grid.ndimension()!=5){
        ERROR("incorrect input dimension");
    }
    int batch_size = rotations.size(0);
    int size = grid.size(1);
    cpu_RotateGrid(rotations.data<float>(), grid.data<float>(), batch_size, size);
}

