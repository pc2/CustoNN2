import tensorflow as tf


from PIL import Image
import numpy as np
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import cv2
from numpy import array

with tf.Graph().as_default() as graph: # Set default graph as graph

           with tf.Session() as sess:
                # Load the graph in graph_def
                print("load graph")

                # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
                with tf.gfile.FastGFile("frozen_quant.pb",'rb') as f:

                                print("Load Image...")
                                # Read the image & get statstics
                                image = scipy.misc.imread('pepper.png')
                                image = image.astype(float)
                                Input_image_shape=image.shape
                                height,width,channels = Input_image_shape
                                image = array(image).reshape(1,224,224,3)
                                print("Plot image...")
                                
                                #scipy.misc.imshow(image)
                                #if image.dtype != tf.float32:
                                #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                                
                                
                                image = np.divide(image,255)
                                image = np.subtract(image, 0.5)
                                image = np.multiply(image, 2.0)
                                # Set FCN graph to the default graph
                                graph_def = tf.GraphDef()
                                graph_def.ParseFromString(f.read())
                                sess.graph.as_default()
                                '''
                                for j in range (224):
                                    for i in range (2):
                                        print("Image value 0 is : " , image[0][i][j][0] , file=open("output.txt", "a"))
                                '''
 
                                
                               
                                # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)

                                tf.import_graph_def(
                                graph_def,
                                input_map=None,
                                return_elements=None,
                                name="",
                                op_dict=None,
                                producer_op_list=None
                                )
                                
                                #Uncomment the following for loop if you want to generate tensor ops and save to a file named layers.txt
                                # Print the name of operations in the session
                                #for op in graph.get_operations():
                                #       print("Operation Name :",op.name ,file=open("layers.txt", "a"))         # Operation name
                                #       print("Tensor Stats :",str(op.values()) , file=open("layers.txt", "a"))      # Tensor name

                                run_until = 'InceptionV1/Logits/Predictions/Reshape_1:0'
                                
                                # Set the following  to 1 when maxpool is inside inception module
                                # This is exception override when Maxpool is inside inception module.
                                exception_override = 0 
                                 
                                # INFERENCE Here
                                l_input = graph.get_tensor_by_name('input:0') # Input Tensor
                                l_output = graph.get_tensor_by_name(run_until) # Output Tensor
                                
                                
                                #the format for run_until is different for maxpools and concat which
                                #appear out of incpetion modules. This is a list to process such tensor
                                #ops differently to print their outputs
                                
                                exception_list = ["concat","MaxPool","AvgPool","Predictions"]
                                
                                
                                 
                                
                                if(run_until.find(exception_list[0]) != -1 or run_until.find(exception_list[1]) != -1  or run_until.find(exception_list[2]) != -1 or  run_until.find(exception_list[3]) != -1) :
                                    exception_layer = 1
                                    print("Exception is {} so truncated file name will be used".format(exception_layer) )
                                else:
                                    exception_layer =  0                                   
                                    print("Exception is {} so normal file name will be used".format(exception_layer) )
                                    
                                if(exception_override == 1 and exception_layer == 1):
                                    exception_layer = 0 
                                
                                
                                if (exception_layer == 0):
                                    inception_model,inception_model_re,inception_level,branch_level,operation_name,extra_info = run_until.split("/")
                                    file_path_info = str(inception_level)+"_"+str(branch_level)+"_"+str(operation_name)
                                elif (exception_layer == 1):
                                    inception_model,inception_model_re,inception_level,operation_name_mit_extra = run_until.split("/") 
                                    operation_name = operation_name_mit_extra.split(":")
                                    branch_level = str("keinbranch")
                                    file_path_info = str(inception_level)+"_"+str(branch_level)+"_"+str(operation_name[0])
                                    
                                    
                                    
                                    
                                    
                                print(inception_level)
                                print(branch_level)
                                print(operation_name)
                                print(extra_info)
                                
                                 
                                print(file_path_info)
                                
                                 
                                #change the path here
                                file_path_xxxx ="D:\Paderborn\ProjectCNNFPGA\Tensorflow_code\TF_Outpts"
                                
                                file_path_NHWC =str(file_path_xxxx)+ "\{}_NHWC.txt".format(file_path_info)
                                file_path_NCHW =str(file_path_xxxx)+ "\{}_NCHW.txt".format(file_path_info)


                                
                                print ("Shape of input : ", tf.shape(l_input))#  , file=open("output.txt", "a"))
                                
                                #initialize_all_variables
                                tf.global_variables_initializer()

                                # Run Googlenet model on single image
                                Session_out = sess.run( l_output, feed_dict ={l_input : image})
                                max = -99999
                                max_ind = 0
 

##############################################################################################################################################################
     
                                height = len(Session_out[0])
                                width  = len(Session_out[0][0])
                                channel_depth  = len(Session_out[0][0][0])
                                
                                print("\nHeight of tensor :{} \nWidth  of tensor :{}\nDepth  of tensor :{}".format(height,width,channel_depth))
                                 

                                
                                myfile_NCHW = open(file_path_NCHW, 'w')
                                for i in range (channel_depth): 
                                    for j in range (height): 
                                        for k in range (width): 
                                     
                                            myfile_NCHW.write(str(Session_out[0][j][k][i]) +'\n')                                
                                     
                                myfile_NCHW.close()  
                                

                                
                                myfile_NHWC = open(file_path_NHWC, 'w')
                                for i in range (height): 
                                    for j in range (width): 
                                        for k in range (channel_depth): 
                                     
                                            myfile_NHWC.write(str(Session_out[0][i][j][k]) +'\n')                                
                                     
                                myfile_NHWC.close()                                       
                                
                                 
                                '''
                                f = open("first_conv.txt","w")
                                print(len(Session_out))
                                f.write(str(Session_out))
                                result = np.where(Session_out[0] == np.amax(Session_out[0]))
                                '''
                                
                                '''
                                print('Returned tuple of arrays :', result)
                                print('List of Indices of maximum element :', result[0])
                                ''' 
                                
                                '''
                                for i in range (1001):
                                    print(str(Session_out[0][i]))
                                    if(Session_out[0][i]>max):
                                        max = Session_out[0][i]
                                        max_ind = i

                                print(str(max_ind))
                                '''