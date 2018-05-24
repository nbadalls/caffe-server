cd /home/zkx/Project/O2N/caffe-master
./build/tools/caffe train \
--solver="models/AMlmdbCdata/AMlmdbCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd+encode_faceNet-20-light2s4-bn/solver.prototxt" \
--gpu="0,1,2,3" \
--snapshot="../asset/snapshot/AMlmdbCdata/AMlmdbCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd+encode_faceNet-20-light2s4-bn/2018-05-22_AMlmdbCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd+encode_faceNet-20-light2s4-bn_zkx_iter_80000.solverstate" \
2>&1 | tee jobs/AMlmdbCdata/AMlmdbCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd+encode_faceNet-20-light2s4-bn/2018-05-23_AMlmdbCdata-b0.35s30_fc_0.35_112x96_b+FaceAdd+encode_faceNet-20-light2s4-bn_zkx.log
