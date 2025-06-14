#include "hello.hpp"

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>

int HP_helloOCL() {
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
	std::ifstream helloWorldFile("src/hello.cl");
	if (!helloWorldFile.is_open()) {
		std::cerr << "ERROR: Failed to open hello.cl\n";
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
	const size_t buf_size = 13;  // Matches kernel output
	char buf[buf_size] = {0};	// Initialize to zeros
	
	cl::Buffer memBuf(
		context, 
		CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 
		buf_size
	);

	// Create kernel with error check
	cl::Kernel kernel(program, "Hello", &build_err);
	if (build_err != CL_SUCCESS) {
		std::cerr << "Kernel creation error: " << build_err << "\n";
		return 1;
	}

	build_err = kernel.setArg(0, memBuf);
	if (build_err != CL_SUCCESS) {
		std::cerr << "Argument setting error: " << build_err << "\n";
		return 1;
	}

	// Execute kernel
	cl::CommandQueue queue(context, device);
	build_err = queue.enqueueNDRangeKernel(
		kernel,
		cl::NullRange,
		cl::NDRange(1), // 1 work item
		cl::NullRange
	);
	
	if (build_err != CL_SUCCESS) {
		std::cerr << "Kernel execution error: " << build_err << "\n";
		return 1;
	}

	// Read results with verification
	build_err = queue.enqueueReadBuffer(
		memBuf, 
		CL_TRUE, 
		0, 
		buf_size, 
		buf
	);
	
	if (build_err != CL_SUCCESS) {
		std::cerr << "Buffer read error: " << build_err << "\n";
		return 1;
	}

	// Print results (raw data verification)
	std::cout << "Output: ";
	for (size_t i = 0; i < buf_size; i++) {
		if (buf[i] == '\0') buf[i] = ' ';  // Replace nulls for visibility
	}
	std::cout.write(buf, buf_size);
	std::cout << "\n";

	return 0;
}
