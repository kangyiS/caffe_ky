#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a leveldb/lmdb\n"
        "Usage:\n"
        "    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
    return 1;
  }

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[1], db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());

  BlobProto sum_blob;
  int count = 0;
  // load first datum
  //Datum datum;
  //datum.ParseFromString(cursor->value());
  AnnotatedDatum anno_datum;
  anno_datum.ParseFromString(cursor->value());
  Datum datum = anno_datum.datum();

  if (DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
    printf("Decoding Datum \n");
  }

  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  printf("Starting Iteration \n");
// process one image per loop
  while (cursor->valid()) {
    //printf("string is : %s \n", cursor->value());
    //Datum datum;
    //datum.ParseFromString(cursor->value());
    AnnotatedDatum anno_datum;
    anno_datum.ParseFromString(cursor->value());
    Datum datum = anno_datum.datum();
    DecodeDatumNative(&datum);
    //printf("datum params are: %d, %d, %d \n", datum.channels(), datum.height(), datum.width());
    const std::string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    //printf("size_in_datum and data_size are: %d, %d \n", size_in_datum, data_size);
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
// if this image has data, then add every pixel
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 1000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
      printf("Processed %d files. \n", count);
    }
    cursor->Next();
  }

  if (count % 1000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
    printf("Processed %d files. \n", count);
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  if (argc == 3) {
    LOG(INFO) << "Write to " << argv[2];
    printf("Write to %s \n", argv[2]);
    WriteProtoToBinaryFile(sum_blob, argv[2]);
  }
  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<float> mean_values(channels, 0.0);
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    printf("mean value channel %d : %f \n", c, mean_values[c] / dim);
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
  printf("This tool requires OpenCV; compile with USE_OPENCV. \n");
#endif  // USE_OPENCV
  return 0;
}
