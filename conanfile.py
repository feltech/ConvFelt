import os

from conan import ConanFile
from conan.tools.scm import Git
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
from conan.tools.files import copy


class ConvFeltRecipe(ConanFile):
    name = "convfelt"
    version = "develop"
    package_type = "library"

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False], "enable_tests": [True, False]}
    default_options = {"shared": False, "fPIC": True, "enable_tests": False}

    exports_sources = "CMakeLists.txt", "src/*", "cmake/*"

    def requirements(self):
        self.requires("adaptivecpp/develop")
        self.requires("onemkl/develop")
        self.requires("eigen/master")

        self.requires("mdspan/0.6.0")

        # Boost TODO(DF): seeing if boost::format works in kernels
        self.requires("boost/1.84.0")

        # Coroutines
        self.requires("andreasbuhr-cppcoro/cci.20230629")

        # Required for CucumberCpp (it bundles it but doesn't install it)
        # TODO(DF): maybe not needed for CucumberCpp 0.7.0
        # self.requires("gtest/1.14.0")

        # Command-line parser
        # - Required for CucumberCpp 0.7.0
        self.requires("tclap/1.2.5")

        # YAML parser
        # - Used to parametrise network structure
        # - Dependency of cucumber-cpp-runner
        self.requires("yaml-cpp/0.8.0")

        # JSON parser
        # - Dependency of CucumberCpp
        self.requires("nlohmann_json/3.11.3")

        # Async networking library extracted from boost.
        # - Dependency of cucumber-cpp-runner
        # - Dependency of CucumberCpp
        self.requires("asio/1.29.0")

        # Pin to boost version such that OpenImageIO supports boost::filesystem without undefined
        # references
        # TODO(DF): may not be necessary if/when libstdc++ can be upgraded i.e. when Clang 17 is in
        #   Conda)
        # self.requires("boost/1.77.0")

        # String formatting. Must override dependency of openimageio or get ambiguous calls to
        # `std::signbit` et al.
        # TODO(DF): may not be necessary if/when libstdc++ can be upgraded i.e. when Clang 17 is in
        #   Conda)
        # self.requires("fmt/7.1.3")
        # self.requires("fmt/9.1.0")

        # Embedded template library - maybe some of these functions (e.g. strings) are usable in
        # kernels?
        self.requires("etl/20.38.10")

        # OpenImageIO image loading/processing library"
        self.requires("openimageio/2.4.7.1")

        # Override to >=v3 so that Imath is used, rather than a fallback (OIIO_USING_IMATH), which
        # works around "error: definition of type 'half' conflicts with typedef of the same name"
        # when CUDA (via OpenSYCL) is also included.
        # self.requires("openexr/3.1.5")

        # range-v3 (whilst waiting for std::range) library
        self.requires("range-v3/0.12.0")

        if bool(self.options.enable_tests):
            self.requires("catch2/3.5.3")
            self.requires("trompeloeil/47")

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

        self.options["openimageio"].with_ffmpeg = False  # Too many transitive deps, e.g. xorg.
        # Required for openimageio
        self.options["boost"].without_thread = False
        self.options["boost"].without_atomic = False
        self.options["boost"].without_chrono = False
        self.options["boost"].without_container = False
        self.options["boost"].without_date_time = False
        self.options["boost"].without_exception = False
        self.options["boost"].without_system = False
        self.options["boost"].without_regex = False

        self.options["fmt"].with_os_api = False
        # self.options["fmt"].header_only = True

    def layout(self):
        cmake_layout(self, src_folder=".")

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.cache_variables["CONVFELT_ENABLE_TESTS"] = bool(self.options.enable_tests)
        tc.cache_variables["CMAKE_VERBOSE_MAKEFILE"] = True
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.builddirs = [os.path.join(self.package_folder)]
