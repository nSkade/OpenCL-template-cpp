#include "vecAdd.hpp"

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>

int HP_vecAdd() {
	// Get platforms
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty()) {
		std::cerr << "ERROR: No OpenCL platforms found\n";
		return 1;
	}

	std::cout << "Found " << platforms.size() << " platform(s)\n";

	cl::Device device;
	bool device_found = false;

	{ // print all platforms
		int i=0;
		for (auto& platform : platforms) {
			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>()
					  << "\nDevices found: " << devices.size() << "\n";
			for (auto& device : devices)
				std::cout << "	device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
		}
	}
	
	// Device selection with better error handling
	for (auto& platform : platforms) {
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		if (!devices.empty()) {
			device = devices.front();
			std::cout << "Using device: "
					  << device.getInfo<CL_DEVICE_NAME>() << "\n";
			device_found = true;
			break;
		}
	}

	if (!device_found) {
		std::cerr << "ERROR: No OpenCL devices found\n";
		return 1;
	}

	// Load kernel source, path relative to project root
	std::ifstream helloWorldFile("src/vecAdd.cl");
	if (!helloWorldFile.is_open()) {
		std::cerr << "ERROR: Failed to open vecAdd.cl\n";
		return 1;
	}
	
	std::string src(
		(std::istreambuf_iterator<char>(helloWorldFile)),
		(std::istreambuf_iterator<char>())
	);

	cl::Program::Sources sources({src});
	cl::Context context(device);
	cl::Program program(context, sources);

	// Build program with error checking
	cl_int build_err = program.build("-cl-std=CL1.2");
	if (build_err != CL_SUCCESS) {
		std::cerr << "Build error:\n" 
				  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		return 1;
	}

	// Set up memory buffers
	const size_t buf_a_size = 13*sizeof(cl_float);
	cl_float buf_a[13];

	const size_t buf_b_size = 13*sizeof(cl_float);
	cl_float buf_b[13];

	for (int i=0;i<13;++i) {
		buf_a[i] = float(i)+1.;
		buf_b[i] = float(i)+1.;
	}
	
	cl::Buffer memBuf_a = cl::Buffer(
		context, 
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
		buf_a_size
	);
	
	cl::Buffer memBuf_b = cl::Buffer(
		context, 
		CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
		buf_b_size
	);

	// output buffer
	const size_t buf_c_size = 13*sizeof(cl_float);
	cl_float buf_c[buf_c_size];

	cl::Buffer memBuf_c = cl::Buffer(
		context, 
		CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
		buf_c_size
	);

	// Create kernel with error check
	cl::Kernel kernel(program, "vector_add", &build_err);
	if (build_err != CL_SUCCESS) {
		std::cerr << "Kernel creation error: " << build_err << "\n";
		return 1;
	}

	build_err = kernel.setArg(0, memBuf_a);
	if (build_err != CL_SUCCESS) {
		std::cerr << "Argument setting error: " << build_err << "\n";
		return 1;
	}
	build_err = kernel.setArg(1, memBuf_b);
	if (build_err != CL_SUCCESS) {
		std::cerr << "Argument setting error: " << build_err << "\n";
		return 1;
	}
	build_err = kernel.setArg(2, memBuf_c);
	if (build_err != CL_SUCCESS) {
		std::cerr << "Argument setting error: " << build_err << "\n";
		return 1;
	}

	// Execute kernel
	cl::CommandQueue queue(context, device);
	build_err = queue.enqueueWriteBuffer(
		memBuf_a,
		CL_TRUE,
		0,
		buf_a_size,
		buf_a
	);
	build_err = queue.enqueueWriteBuffer(
		memBuf_b,
		CL_TRUE,
		0,
		buf_b_size,
		buf_b
	);
	if (build_err != CL_SUCCESS) {
		std::cerr << "WriteBuffer error: " << build_err << "\n";
		return 1;
	}

	build_err = queue.enqueueNDRangeKernel(
		kernel,
		cl::NullRange,
		cl::NDRange(13), // global thread count
		cl::NullRange    // local thread count (per work group)
		                 // this is automatically 13 (if smaller than HW work group size)
		// -> workgroup count is global thread count / local thread count
	);
	
	if (build_err != CL_SUCCESS) {
		std::cerr << "Kernel execution error: " << build_err << "\n";
		return 1;
	}

	// Read results with verification
	build_err = queue.enqueueReadBuffer(
		memBuf_c,
		CL_TRUE,
		0,
		buf_c_size,
		buf_c
	);
	
	if (build_err != CL_SUCCESS) {
		std::cerr << "Buffer read error: " << build_err << "\n";
		return 1;
	}

	// Print results (raw data verification)
	std::cout << "Output: \n";
	for (size_t i = 0; i < 13; i++) {
		std::cout << buf_c[i] << ", \n";
	}
	std::cout << "\n";

	return 0;
}
