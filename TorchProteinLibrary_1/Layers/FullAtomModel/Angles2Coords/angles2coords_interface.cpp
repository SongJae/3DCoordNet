
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/FullAtomModel/Angles2Coords/angles2coords_interface.h>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Layers/FullAtomModel/cConformation.h>
#include <iostream>
#include <string>
#include </workspace/Molecule3D/TorchProteinLibrary_1/Math/nUtil.h>

void Angles2Coords_forward(     torch::Tensor sequences,
                                torch::Tensor input_angles, 
                                torch::Tensor output_coords,
                                torch::Tensor res_names,
                                torch::Tensor atom_names
                        ){
    bool add_terminal = false;
    CHECK_CPU_INPUT_TYPE(sequences, torch::kByte);
    CHECK_CPU_INPUT(input_angles);
    CHECK_CPU_INPUT(output_coords);
    CHECK_CPU_INPUT_TYPE(res_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(atom_names, torch::kByte);

    if(input_angles.ndimension() != 3){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = input_angles.sizes()[0];
             
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
                
        torch::Tensor single_sequence = sequences[i];
        torch::Tensor single_atom_names = atom_names[i];
        torch::Tensor single_res_names = res_names[i];
        torch::Tensor single_angles = input_angles[i];
        torch::Tensor single_coords = output_coords[i];
        
        std::string seq = StringUtil::tensor2String(single_sequence);
        
        int length = single_angles.sizes()[1];
        int num_atoms = ProtUtil::getNumAtoms(seq, add_terminal);
        
        if( single_coords.sizes()[0]<3*num_atoms){
            ERROR("incorrect coordinates tensor length");
        }
        
        if( length<seq.length() || single_angles.sizes()[0]<8 ){
            ERROR("incorrect angles tensor length");
        }
        
        if( single_res_names.sizes()[0]<seq.length() ){
            ERROR("incorrect res names tensor length");
        }
        
        if( single_atom_names.sizes()[0]<seq.length() ){
            ERROR("incorrect atom names tensor length");
        }
        torch::Tensor dummy_grad = torch::zeros_like(single_angles);
        AT_DISPATCH_FLOATING_TYPES(single_angles.type(), "cConformation", ([&] {
            cConformation<scalar_t> conf( seq, single_angles.data<scalar_t>(), dummy_grad.data<scalar_t>(), length, single_coords.data<scalar_t>());
            for(int j=0; j<conf.groups.size(); j++){
                for(int k=0; k<conf.groups[j]->atomNames.size(); k++){
                    int idx = conf.groups[j]->atomIndexes[k];
                    torch::Tensor single_atom_name = single_atom_names[idx];
                    torch::Tensor single_res_name = single_res_names[idx];
                    StringUtil::string2Tensor(ProtUtil::convertRes1to3(conf.groups[j]->residueName), single_res_name);
                    StringUtil::string2Tensor(conf.groups[j]->atomNames[k], single_atom_name);
                }
            }                    
        }));
        // cConformation<double> conf( seq, single_angles.data<double>(), dummy_grad.data<double>(),
        //                     length, single_coords.data<double>());
        //Output atom names and residue names
        
    }
}

void Angles2Coords_backward(    torch::Tensor grad_atoms,
                                torch::Tensor grad_angles,
                                torch::Tensor sequences,
                                torch::Tensor input_angles
                        ){
    bool add_terminal = false;
    CHECK_CPU_INPUT_TYPE(sequences, torch::kByte);
    CHECK_CPU_INPUT(input_angles);
    CHECK_CPU_INPUT(grad_atoms);
    CHECK_CPU_INPUT(grad_angles);
    if(input_angles.ndimension() != 3){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = input_angles.sizes()[0];

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        
        torch::Tensor single_sequence = sequences[i];
        torch::Tensor single_angles = input_angles[i];
        torch::Tensor single_grad_angles = grad_angles[i];
        torch::Tensor single_grad_atoms = grad_atoms[i];
        
        std::string seq = StringUtil::tensor2String(single_sequence);
                
        uint length = single_angles.sizes()[1];
        int num_atoms = ProtUtil::getNumAtoms(seq, add_terminal);
        
        torch::Tensor dummy_coords = torch::zeros({3*num_atoms}, torch::TensorOptions().dtype(grad_atoms.dtype()));
        AT_DISPATCH_FLOATING_TYPES(single_angles.type(), "cConformation", ([&] {
            cConformation<scalar_t> conf(seq, single_angles.data<scalar_t>(), single_grad_angles.data<scalar_t>(),length, dummy_coords.data<scalar_t>());
            conf.backward(conf.root, single_grad_atoms.data<scalar_t>());
        }));
        // cConformation<double> conf( seq, single_angles.data<double>(), single_grad_angles.data<double>(),
        //                     length, dummy_coords.data<double>());
        // conf.backward(conf.root, single_grad_atoms.data<double>());
    }

}

void Angles2Coords_save(    const char* sequence,
                            torch::Tensor input_angles, 
                            const char* output_filename,
                            const char mode
                        ){
    bool add_terminal = false;
    if(input_angles.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    std::string aa(sequence);
    uint length = aa.length();
    int num_atoms = ProtUtil::getNumAtoms(aa, add_terminal);
    torch::Tensor dummy_grad = torch::zeros_like(input_angles);
    torch::Tensor dummy_coords = torch::zeros({3*num_atoms}, torch::TensorOptions().dtype(input_angles.dtype()));
    AT_DISPATCH_FLOATING_TYPES(input_angles.type(), "cConformation.save", ([&]{
        cConformation<double> conf( aa, input_angles.data<double>(), dummy_grad.data<double>(), 
                        length, dummy_coords.data<double>());
        conf.save(std::string(output_filename), mode);
    }));
    
}

int getSeqNumAtoms( const char *sequence){
    bool add_terminal = false;
    std::string seq(sequence);
    int num_atoms = ProtUtil::getNumAtoms(seq, add_terminal);
    return num_atoms;
}
