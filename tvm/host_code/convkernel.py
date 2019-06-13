import tvm

tgt_host="llvm"
tgt="aocl_sw_emu"




#n=G_NUMBER_OF_FILERS=32
#m=G_NUMBER_OF_FILTER_ROWS=5
#p=G_NUMBER_OF_FILTER_COLS=5


G_NUMBER_OF_FILTER_ROWS=tvm.var('G_NUMBER_OF_FILTER_ROWS')
G_NUMBER_OF_FILTER_COLS=tvm.var('G_NUMBER_OF_FILTER_COLS')
G_NUMBER_OF_FILERS=tvm.var('G_NUMBER_OF_FILERS')


cnnWeight=tvm.placeholder((G_NUMBER_OF_FILTER_ROWS,G_NUMBER_OF_FILTER_COLS,G_NUMBER_OF_FILERS),'cnnWeight')
cnnBias=tvm.placeholder((G_NUMBER_OF_FILERS),'cnnBias')



#cnnWeightLocal=tvm.compute(cnnWeight.shape, lambda i: cnnWeight[i], name='cnnWeightLocal')
#cnnBiasLocal=tvm.compute(cnnBias.shape, lambda k: G_NUMBER_OF_FILERS[k], name="cnnBiasLocal")



#int numberOfTotalPixels = G_NUMBER_OF_IMAGES*G_NUMBER_OF_IMAGE_ROWS*G_NUMBER_OF_IMAGE_COLS;
#int numberOfImagePixels = G_NUMBER_OF_IMAGE_ROWS*G_NUMBER_OF_IMAGE_COLS;

numberofTotalPixels=tvm.var('numberofTotalPixels',int)
numberOfImagePixels=tvm.var('numberOfImagePixels',int)



G_NUMBER_OF_IMAGES=tvm.var('G_NUMBER_OF_IMAGES')
inX=tvm.var('inX',int)
inY=tvm.var('inY',int)
G_NUMBER_OF_CONV_OUT_ROWS=tvm.var('G_NUMBER_OF_CONV_OUT_ROWS')
G_NUMBER_OF_CONV_OUT_COLS=tvm.var('G_NUMBER_OF_CONV_OUT_COLS')

conv=tvm.const(0,int)

outRowIndex=tvm.placeholder((inX),'outRowIndex')
outColIndex=tvm.placeholder((inY),'outColIndex')

#cnnbias
#conv = cnnBiasLocal[filterNumber]

ConvFilterRowIndex=tvm.compute(inX.shape, lambda i,j,k,l,m:(filterNumber[i]*G_NUMBER_OF_FILTER_ROWS[j]*G_NUMBER_OF_FILTER_COLS[k])+(filterRowIndex[l]*G_NUMBER_OF_FILTER_COLS[m])
ConvImgRowIndex=tvm.compute(inX.shape, lambda i,j,k,l,m,n:((imgIndex[i]*G_NUMBER_OF_IMAGE_ROWS[j]*G_NUMBER_OF_IMAGE_COLS[k])+(inX[l]*G_NUMBER_OF_IMAGE_COLS[m])+inY


#ConvImgRowIndex=tvm.compute(
























































