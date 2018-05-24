cd /home/zkx/Project/O2N/caffe-master
./build/tools/caffe train \
--solver="models/AMImageCdata/AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet-bn/solver.prototxt" \
--gpu="0,1,2,3" \
--weights="/home/zkx/Project/O2N/DepthwiseConvolution-master/MobileFaceNet-master/face_snapshot/MobileFaceNet.caffemodel" \
2>&1 | tee jobs/AMImageCdata/AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet-bn/2018-05-23_AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet-bn_zkx.log
