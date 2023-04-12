#include <torch/extension.h>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/Volume/TypedCoords2Volume/typedcoords2volume_interface.h>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/Volume/Volume2Xplor/volume2xplor_interface.h>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/Volume/Select/select_interface.h>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/Volume/VolumeConvolution/volumeConvolution_interface.h>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/Volume/VolumeRotation/volumeRotation_interface.h>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/Volume/VolumeRMSD/volumeRMSD_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("TypedCoords2Volume_forward", &TypedCoords2Volume_forward, "TypedCoords2Volume forward");
    m.def("TypedCoords2Volume_backward", &TypedCoords2Volume_backward, "TypedCoords2Volume backward");
    m.def("Volume2Xplor", &Volume2Xplor, "Save 3D volume as xplor file");
    m.def("SelectVolume_forward", &SelectVolume_forward, "Select feature columns from volume at coordinates");
    m.def("VolumeConvolution_forward", &VolumeConvolution_forward, "VolumeConvolution forward");
    m.def("VolumeConvolution_backward", &VolumeConvolution_backward, "VolumeConvolution backward");
    m.def("VolumeGenGrid", &VolumeGenGrid, "Volume generate rotated grid");
    m.def("VolumeGenRMSD", &VolumeGenRMSD, "Generate RMSD on the circular grid of displacements");
}
