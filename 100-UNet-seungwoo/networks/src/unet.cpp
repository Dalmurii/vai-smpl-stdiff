#include "../include/unet.hpp"

using namespace networks;
using namespace std;


;
Unet::Unet(Device& device, uint32_t  height, uint32_t width, uint32_t channel, uint32_t numInputs, uint32_t numOutputs)
	: NeuralNet(device, numInputs, numOutputs)
	,device_(device), height_(height), width_(width), channel_(channel), numInputs_(numInputs), numOutputs_(numOutputs)
{
	cout << "Unet Init Done" << endl;
}