cd /home/kangyi/caffe
./build/tools/caffe train \
--solver="examples/MobileSSD/MobileSSD_VOC0712/prototxt/solver_train.prototxt" \
--weights="examples/MobileSSD/MobileSSD_VOC0712/model/mobilenet_iter_73000.caffemodel" \
--gpu 0 2>&1 | tee examples/MobileSSD/MobileSSD_VOC0712/log/MobileSSD_VOC0712.log
