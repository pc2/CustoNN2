# Image Classification Sample

This topic demonstrates how to run the Image Classification sample application, which performs 
inference using image classification networks such as AlexNet and GoogLeNet.

## How to execute :
- set a temporary variable to store the path of the Intermediate Representation 
 `export IR='/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation'`

- Execute the test_plugin application  
 `./test_plugin -m $IR/lenet_iter_10000.xml -i $IR/three.png`  
 **format:**  ./test_plugin -m <model_path> -i <images_path>

