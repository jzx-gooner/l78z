#include "simple_yolo.hpp"
#include "deepsort.hpp"
#if defined(_WIN32)
#include <Windows.h>
#include <wingdi.h>
#include <Shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "gdi32.lib")
#undef min
#undef max
#else
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>
#endif

#define USE_DEEPSORT true;

using namespace std;

class MotionFilter
{
public:
    MotionFilter()
    {
        location_.left = location_.top = location_.right = location_.bottom = 0;
    }

    void missed()
    {
        init_ = false;
    }

    void update(const DeepSORT::Box &box)
    {

        const float a[] = {box.left, box.top, box.right, box.bottom};
        const float b[] = {location_.left, location_.top, location_.right, location_.bottom};

        if (!init_)
        {
            init_ = true;
            location_ = box;
            return;
        }

        float v[4];
        for (int i = 0; i < 4; ++i)
            v[i] = a[i] * 0.6 + b[i] * 0.4;

        location_.left = v[0];
        location_.top = v[1];
        location_.right = v[2];
        location_.bottom = v[3];
    }

    DeepSORT::Box predict()
    {
        return location_;
    }

private:
    DeepSORT::Box location_;
    bool init_ = false;
};

static const char *cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i)
    {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    case 5:
        r = v;
        g = p;
        b = q;
        break;
    default:
        r = 1;
        g = 1;
        b = 1;
        break;
    }
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    ;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

static bool exists(const string &path)
{

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

static string get_file_name(const string &path, bool include_suffix)
{

    if (path.empty())
        return "";

    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;

    //include suffix
    if (include_suffix)
        return path.substr(p);

    int u = path.rfind('.');
    if (u == -1)
        return path.substr(p);

    if (u <= p)
        u = path.size();
    return path.substr(p, u - p);
}

static double timestamp_now_float()
{
    return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

bool requires_model(const string &name)
{

    auto onnx_file = cv::format("%s_dynamic.onnx", name.c_str());
    if (!exists(onnx_file))
    {
        printf("Auto download %s\n", onnx_file.c_str());
        system(cv::format("wget http://zifuture.com:1556/fs/25.shared/%s", onnx_file.c_str()).c_str());
    }

    bool isexists = exists(onnx_file);
    if (!isexists)
    {
        printf("Download %s failed\n", onnx_file.c_str());
    }
    return isexists;
}

// cv::Mat show_result(cv::Mat &image, std::vector<SimpleYolo::Box> &objs)
// {
//     for (auto &obj : objs)
//     {
//         uint8_t b, g, r;
//         tie(b, g, r) = random_color(obj.class_label);
//         cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

//         auto name = cocolabels[obj.class_label];
//         auto caption = cv::format("%s %.2f", name, obj.confidence);
//         int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
//         cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
//         cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
//     }
//     cv::imshow("debug", image);
//     cv::waitKey(1);
// }

void test_yolov5_deepsort()
{
    //1.load model build yolo class
    printf("TRTVersion: %s\n", SimpleYolo::trt_version());
    int device_id = 0;
    string model = "yolox_s";
    auto type = SimpleYolo::Type::X;
    auto mode = SimpleYolo::Mode::FP32;
    string onnx_file = cv::format("%s_dynamic.onnx", model.c_str());
    string model_file = cv::format("%s_dynamic.%s.trtmodel", model.c_str(), SimpleYolo::mode_string(mode));
    SimpleYolo::set_device(device_id);

    if (!requires_model(model))
    {
        printf("Download failed\n");
        return;
    }
    if (!exists(model_file) && !SimpleYolo::compile(mode, type, 6, onnx_file, model_file, 1 << 30, "inference"))
    {
        printf("Compile failed\n");
        return;
    }

    float confidence_threshold = 0.4f;
    float nms_threshold = 0.5f;
    auto yolo = SimpleYolo::create_infer(model_file, type, device_id, confidence_threshold, nms_threshold);
    if (yolo == nullptr)
    {
        printf("Yolo is nullptr\n");
        return;
    }
   
    //0.初始化追踪器
    auto config = DeepSORT::TrackerConfig();
    config.has_feature = false;
    config.max_age = 150;
    config.nbuckets = 150;
    config.distance_threshold = 8000.0f;

    config.set_per_frame_motion({0.05, 0.02, 0.1, 0.02,
                                 0.08, 0.02, 0.1, 0.02});
    auto tracker = DeepSORT::create_tracker(config);

     //1.get rtsp opencv自带方法
    //todo:需要优化
    cv::VideoCapture cap;
    cap.open("rtsp://admin:lnint521@192.168.212.22:554/cam/realmonitor?channel=1&subtype=1");
    //cap.open("/home/jzx/sgai_yolov5/sgai_yolov5_infer/workspace/inference/test.mp4");
    if (!cap.isOpened())
    {
        printf("open rtsp failed,open the video insted\n");
        cap.open("/home/jzx/sgai_yolov5/sgai_yolov5_infer/workspace/inference/test.mp4");
    }
    cv::Mat image;
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 768);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1024);


    unordered_map<int, MotionFilter> MotionFilter;

    while (1)
    {
        cap >> image;
        if (image.empty())
        {
            break;
        }
        auto det_objs = yolo->commit(image).get();
        // cout<<"det objets size : "<<to_string(det_objs.size())<<std::endl;
        vector<DeepSORT::Box> boxes;
        for (int i = 0; i < det_objs.size(); ++i)
        {
            auto &det_obj = det_objs[i];
            //std::cout<<to_string(det_obj.class_label)<<std::endl;
            if (det_obj.class_label == 0)
            { //只有在检测是person的时候才更新追踪
                auto track_box = DeepSORT::convert_to_box(det_obj);
                //track_box.feature = det_obj.feature;
                boxes.emplace_back(std::move(track_box));
            }
        }
        //debug
        // show_result(image, det_objs);

        tracker->update(boxes);

        auto final_objects = tracker->get_objects();
        for (int i = 0; i < final_objects.size(); ++i)
        {
            std::cout << to_string(i) << std::endl;
            auto &obj = final_objects[i];
            auto &filter = MotionFilter[obj->id()];
            if (obj->time_since_update() == 0 && obj->state() == DeepSORT::State::Confirmed)
            {
                uint8_t b, g, r;
                tie(b, g, r) = random_color(obj->id());
                auto loaction = obj->last_position();
                filter.update(loaction);
                loaction = filter.predict();
                cv::rectangle(image, cv::Point(loaction.left, loaction.top), cv::Point(loaction.right, loaction.bottom), cv::Scalar(b, g, r), 5);
                auto name = cocolabels[0]; //loaction.class_label
                auto caption = cv::format("%s %.2f", name, loaction.confidence);
                auto id = obj->id();
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(image, cv::Point(loaction.left - 3, loaction.top - 33), cv::Point(loaction.left + width, loaction.top), cv::Scalar(b, g, r), -1);
                //cv::putText(image, caption, cv::Point(loaction.left, loaction.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
                cv::putText(image, to_string(id), cv::Point(loaction.left, loaction.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }
            else
            {
                filter.missed();
            }
        }

        cv::imshow("rtsp_inference_by_yolov5+deepsort", image);
        cv::waitKey(1);
    }
}



void test_yolov5_rtsp()
{
    //1.load model build yolo class
    printf("TRTVersion: %s\n", SimpleYolo::trt_version());
    int device_id = 0;
    string model = "yolox_s";
    auto type = SimpleYolo::Type::X;
    auto mode = SimpleYolo::Mode::FP32;
    string onnx_file = cv::format("%s_dynamic.onnx", model.c_str());
    string model_file = cv::format("%s_dynamic.%s.trtmodel", model.c_str(), SimpleYolo::mode_string(mode));
    SimpleYolo::set_device(device_id);

    if (!requires_model(model))
    {
        printf("Download failed\n");
        return;
    }
    if (!exists(model_file) && !SimpleYolo::compile(mode, type, 6, onnx_file, model_file, 1 << 30, "inference"))
    {
        printf("Compile failed\n");
        return;
    }

    float confidence_threshold = 0.4f;
    float nms_threshold = 0.5f;
    auto yolo = SimpleYolo::create_infer(model_file, type, device_id, confidence_threshold, nms_threshold);
    if (yolo == nullptr)
    {
        printf("Yolo is nullptr\n");
        return;
    }
   


     //1.get rtsp opencv自带方法
    //todo:需要优化
    cv::VideoCapture cap;
    cap.open("rtsp://admin:lnint521@192.168.212.22:554/cam/realmonitor?channel=1&subtype=1");
    if (!cap.isOpened())
    {
        printf("open rtsp failed,open the video insted\n");
        cap.open("/home/jzx/sgai_yolov5/sgai_yolov5_infer/workspace/inference/test.mp4");
    }
    cv::Mat image;
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 768);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1024);


    while (1)
    {
        cap >> image;
        if (image.empty())
        {
            break;
        }
        auto det_objs = yolo->commit(image).get();
        // cout<<"det objets size : "<<to_string(det_objs.size())<<std::endl;
     
        for (auto &obj : det_objs)
        {
            uint8_t b, g, r;
            tie(b, g, r) = random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);
            auto name = cocolabels[obj.class_label];
            auto caption = cv::format("%s %.2f", name, obj.confidence);
            int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        cv::imshow("rtsp_inference_by_yolov5", image);
        cv::waitKey(1);
    }
}

int main()
{
    test_yolov5_deepsort();
    //test_yolov5_rtsp();
    return 0;
}