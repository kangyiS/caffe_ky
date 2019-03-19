cd /home/kangyi/caffe
./build/tools/caffe train \
--solver="examples/ssd/ssd_RMinfantry/prototxt/solver.prototxt" \
--weights="examples/ssd/ssd_RMinfantry/model/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee examples/ssd/ssd_RMinfantry/log/ssd_RMinfantry.log
