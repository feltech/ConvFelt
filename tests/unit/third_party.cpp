#include <catch2/catch.hpp>

#define VIENNACL_WITH_OPENCL
#include <viennacl/device_specific/builtin_database/common.hpp>
#include <viennacl/vector.hpp>

#include <Eigen/Eigen>

SCENARIO("Loading ViennaCL")
{
	/**
	 *  Retrieve the platforms and iterate:
	 **/
	std::vector<viennacl::ocl::platform> platforms = viennacl::ocl::get_platforms();

	REQUIRE(!platforms.empty());

	bool is_first_element = true;
	for (auto & platform : platforms)
	{
		std::vector<viennacl::ocl::device> devices = platform.devices(CL_DEVICE_TYPE_ALL);

		/**
		 *  Print some platform information
		 **/
		std::cout << "# =========================================" << std::endl;
		std::cout << "#         Platform Information             " << std::endl;
		std::cout << "# =========================================" << std::endl;

		std::cout << "#" << std::endl;
		std::cout << "# Vendor and version: " << platform.info() << std::endl;
		std::cout << "#" << std::endl;

		if (is_first_element)
		{
			std::cout << "# ViennaCL uses this OpenCL platform by default." << std::endl;
			is_first_element = false;
		}

		/*
		 * Traverse the devices and print all information available using the convenience member
		 * function full_info():
		 */
		std::cout << "# " << std::endl;
		std::cout << "# Available Devices: " << std::endl;
		std::cout << "# " << std::endl;
		for (auto & device : devices)
		{
			std::cout << std::endl;

			std::cout << "  -----------------------------------------" << std::endl;
			std::cout << device.full_info();
			std::cout << "ViennaCL Device Architecture:  " << device.architecture_family()
					  << std::endl;
			std::cout << "ViennaCL Database Mapped Name: "
					  << viennacl::device_specific::builtin_database::get_mapped_device_name(
							 device.name(), device.vendor_id())
					  << std::endl;
			std::cout << "  -----------------------------------------" << std::endl;
		}
		std::cout << std::endl;
		std::cout << "###########################################" << std::endl;
		std::cout << std::endl;
	}
}

SCENARIO("Using ViennaCL with Eigen")
{
	Eigen::VectorXd lhs(100);
	lhs.setOnes();
	Eigen::VectorXd rhs(100);
	rhs.setOnes();
	Eigen::VectorXd result(100);
	result.setZero();

	viennacl::vector<double> vcl_lhs(100);
	viennacl::vector<double> vcl_rhs(100);
	viennacl::vector<double> vcl_result(100);

	viennacl::copy(lhs.begin(), lhs.end(), vcl_lhs.begin());
	viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());

	vcl_result = vcl_lhs + vcl_rhs;

	viennacl::copy(vcl_result.begin(), vcl_result.end(), result.begin());

	CHECK(result == Eigen::VectorXd::Constant(100, 2));
}