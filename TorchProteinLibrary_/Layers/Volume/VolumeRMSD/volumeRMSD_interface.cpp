#include "volumeRMSD_interface.h"
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/Volume/VolumeRMSD.h>
#include <iostream>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Math/cVector3.h>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Math/cMatrix33.h>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Math/nUtil.h>

void VolumeGenRMSD( torch::Tensor coords,
                    torch::Tensor num_atoms,
                    torch::Tensor R0,
                    torch::Tensor R1,
                    torch::Tensor T0,
                    torch::Tensor translation_volume,
                    float resolution){
    CHECK_CPU_INPUT_TYPE(coords, torch::kDouble);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    CHECK_CPU_INPUT_TYPE(R0, torch::kDouble);
    CHECK_CPU_INPUT_TYPE(R1, torch::kDouble);
    CHECK_CPU_INPUT_TYPE(T0, torch::kDouble);
    CHECK_GPU_INPUT_TYPE(translation_volume, torch::kFloat);
    if(translation_volume.ndimension()!=4){
        ERROR("incorrect input dimension");
    }

    int batch_size = coords.size(0);
    auto num_atoms_acc = num_atoms.accessor<int,1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_volume = translation_volume[i];
        torch::Tensor single_coords = coords[i];
        torch::Tensor single_R0 = R0[i];
        torch::Tensor single_R1 = R1[i];
        torch::Tensor single_T0 = T0[i];
        auto R0_acc = single_R0.accessor<double,2>();
        auto R1_acc = single_R1.accessor<double,2>();
        auto T0_acc = single_T0.accessor<double,1>();        

        //Computing additive term: 2/N (delta - R0*R1)X
        cMatrix33<double> X; X.setZero();
        double rmsd2R = 0.0;
        for(int j=0; j<num_atoms_acc[i]; j++){
            cVector3<double> r(single_coords.data<double>()+3*j);
            X(0,0) += r.v[0]*r.v[0];
            X(0,1) += r.v[0]*r.v[1];
            X(0,2) += r.v[0]*r.v[2];
            X(1,0) += r.v[1]*r.v[0];
            X(1,1) += r.v[1]*r.v[1];
            X(1,2) += r.v[1]*r.v[2];
            X(2,0) += r.v[2]*r.v[0];
            X(2,1) += r.v[2]*r.v[1];
            X(2,2) += r.v[2]*r.v[2];
        }
        for(int k=0; k<3; k++){
            for(int l=0; l<3; l++){
                double Rsum = 0.0;
                double deltakl = 0.0;
                for(int m=0; m<3; m++){
                    Rsum += R0_acc[m][k] * R1_acc[m][l];
                }
                if(k==l)
                    deltakl = 1.0;
                
                rmsd2R += (deltakl - Rsum)*X(k,l);
        }}
        rmsd2R *= 2.0/double(num_atoms_acc[i]);

        
        //Computing multiplicative term delta: 2.0*(R1 - R2)C
        cVector3<double> centroid; centroid.setZero();
        cVector3<double> delta; delta.setZero();        

        for(int j=0; j<num_atoms_acc[i]; j++){
            centroid += cVector3<double>(single_coords.data<double>()+3*j);
        }
        centroid /= double(num_atoms_acc[i]);

        for(int k=0; k<3; k++){
            for(int m=0; m<3; m++){
                delta.v[k] += 2.0*(R0_acc[k][m] - R1_acc[k][m])*centroid.v[m];
            }
        }

        //Filling volume with RMSD values
        gpu_VolumeRMSD( single_volume.data<float>(), 
                        rmsd2R,  
                        T0_acc[0], T0_acc[1], T0_acc[2],
                        delta.v[0], delta.v[1], delta.v[2], 
                        single_volume.size(0), resolution);

    }
}


