cd /home/zkx/Project/O2N/caffe-master
./build/tools/caffe train \
--solver="models/AdditMarginLmdbImage/AdditMarginLmdbImage-b0.35s30_fc_0.35_112x96_faceNet-20-light2s4-bn/solver.prototxt" \
--gpu="0,1,2,3" \
--snapshot="../asset/snapshot/AdditMarginLmdbImage/AdditMarginLmdbImage-b0.35s30_fc_0.35_112x96_faceNet-20-light2s4-bn/2018-05-22_AdditMarginLmdbImage-b0.35s30_fc_0.35_112x96_faceNet-20-light2s4-bn_zkx_iter_75265.solverstate" \
2>&1 | tee jobs/AdditMarginLmdbImage/AdditMarginLmdbImage-b0.35s30_fc_0.35_112x96_faceNet-20-light2s4-bn/2018-05-22_AdditMarginLmdbImage-b0.35s30_fc_0.35_112x96_faceNet-20-light2s4-bn_zkx.log
