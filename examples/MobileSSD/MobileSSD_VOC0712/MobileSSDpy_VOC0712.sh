cd /home/kangyi/caffe
./build/tools/caffe train \
--solver="examples/MobileSSD/MobileSSDpy_VOC0712/prototxt/solver.prototxt" \
--weights="examples/MobileSSD/MobileSSDpy_VOC0712/model/mobilenet_iter_73000.caffemodel" \
--gpu 0 2>&1 | tee examples/MobileSSD/MobileSSDpy_VOC0712/log/MobileSSDpy_VOC0712.log
