cd /home/zkx/Project/O2N/caffe-master
./build/tools/caffe train \
--solver="models/AMImageMeanCdata/AMImageMeanCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet/solver.prototxt" \
--gpu="0,1,2,3" \
--weights="/home/zkx/Project/O2N/DepthwiseConvolution-master/MobileFaceNet-master/face_snapshot/MobileFaceNet.caffemodel" \
2>&1 | tee jobs/AMImageMeanCdata/AMImageMeanCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet/2018-05-23_AMImageMeanCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet_zkx.log
