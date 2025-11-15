#include "main.hpp"

using namespace std;
using namespace networks;

template<uint32_t Channels>
auto readImage(const char* filename)
{
    int w, h, c0, c = Channels;
    std::vector<uint8_t> srcImage;

    if (uint8_t* input = stbi_load(filename, &w, &h, &c0, c))
    {
        srcImage.assign(input, input + w * h * c);
        stbi_image_free(input);
    }
    else
    {
        printf(stbi_failure_reason());
        fflush(stdout);
        throw;
    }

    return std::make_tuple(srcImage, (uint32_t)w, (uint32_t)h);
}

int main()
{
    Device device = VulkanApp::get().device();

    constexpr uint32_t channels = 3;

    auto [srcImage, width, height] = readImage<channels>("/data/0.png");
    _ASSERT(width * height * channels == srcImage.size());

    Unet::u_ptr net = make_unique<Unet>(device, height, width, channels);

    return 0;
}