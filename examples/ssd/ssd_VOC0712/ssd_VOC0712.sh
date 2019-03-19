cd /home/kangyi/caffe
./build/tools/caffe train \
--solver="examples/ssd/ssd_VOC0712/prototxt/solver.prototxt" \
--snapshot="examples/ssd/ssd_VOC0712/model/ssd_VOC0712_iter_120000.solverstate" \
--gpu 0 2>&1 | tee examples/ssd/ssd_VOC0712/log/ssd_VOC0712.log
