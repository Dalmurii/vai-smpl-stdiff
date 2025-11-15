static const char* src_relu = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main() 
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = max(in0[o], 0.0f);
})";

static const char* src_flatten = R"(
#version 450


void main()
{
    
})";

static const char* src_upconv = R"(
#version 450


void main()
{
    
})";