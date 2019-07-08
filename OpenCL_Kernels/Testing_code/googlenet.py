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

                                # Print the name of operations in the session
                                for op in graph.get_operations():
                                        print("Operation Name :",op.name ,file=open("layers.txt", "a"))         # Operation name
                                        print("Tensor Stats :",str(op.values()) , file=open("layers.txt", "a"))      # Tensor name

                                # INFERENCE Here
                                l_input = graph.get_tensor_by_name('input:0') # Input Tensor
                                l_output = graph.get_tensor_by_name('InceptionV1/InceptionV1/Mixed_3c/concat:0') # Output Tensor
                                

                                
                                print ("Shape of input : ", tf.shape(l_input))#  , file=open("output.txt", "a"))
                                #initialize_all_variables
                                tf.global_variables_initializer()

                                # Run Kitty model on single image
                                Session_out = sess.run( l_output, feed_dict ={l_input : image})
                                max = -99999
                                max_ind = 0
                                
                                #for filter_val in range(256) :
                                
                                for i in range (28):                                     
                                #for j in range(28) :        
                                    print(str(Session_out[0][0][i][0]) , file=open("output.txt", "a"))
                                     
                                  
                                        
                                
                                 
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