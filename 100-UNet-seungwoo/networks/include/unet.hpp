#pragma once

#include <iostream>

#include "../../library/neuralNet.h"
#include "../../library/vulkanApp.h"

namespace networks 
{
	class Unet : public NeuralNet
	{
	private:
		Device				device_;
		uint32_t			numInputs_;
		uint32_t			numOutputs_; 

		uint32_t			height_;
		uint32_t			width_;
		uint32_t			channel_;

	public:
		using s_ptr = std::shared_ptr<Unet>;
		using u_ptr = std::unique_ptr<Unet>;

		Unet(Device& device, uint32_t  height, uint32_t width, uint32_t channel, uint32_t numInputs = 1, uint32_t numOutputs = 1);
		~Unet() = default;

		void operator()()
		{
			(*this)();
		}

		bool CreateNetwork();

		Tensor eval_unet(const std::vector<float>& srcImage, const JsonParser& json);
	};
}
