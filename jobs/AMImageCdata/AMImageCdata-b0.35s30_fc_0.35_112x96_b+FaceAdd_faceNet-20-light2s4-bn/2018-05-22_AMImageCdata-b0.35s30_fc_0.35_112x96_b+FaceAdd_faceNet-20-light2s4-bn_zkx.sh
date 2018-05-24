cd /home/zkx/Project/O2N/caffe-master
./build/tools/caffe train \
--solver="models/AMImageCdata/AMImageCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd_faceNet-20-light2s4-bn/solver.prototxt" \
--gpu="0,1,2,3" \
--snapshot="../asset/snapshot/AMImageCdata/AMImageCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd_faceNet-20-light2s4-bn/2018-05-22_AMImageCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd_faceNet-20-light2s4-bn_zkx_iter_1062.solverstate" \
2>&1 | tee jobs/AMImageCdata/AMImageCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd_faceNet-20-light2s4-bn/2018-05-22_AMImageCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd_faceNet-20-light2s4-bn_zkx.log
