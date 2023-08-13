/**
 * Taken and modified from CucumberCpp main.
 */
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>
#include <boost/process.hpp>
#include <boost/program_options.hpp>
#include <boost/scope_exit.hpp>
#include <cucumber-cpp/internal/CukeEngineImpl.hpp>
#include <cucumber-cpp/internal/connectors/wire/WireProtocol.hpp>
#include <cucumber-cpp/internal/connectors/wire/WireServer.hpp>

namespace
{

struct WireServer
{
	std::string const host;
	int const port;
	std::string const unixPath;
	bool const verbose;

	cucumber::internal::CukeEngineImpl cukeEngine{};
	cucumber::internal::JsonSpiritWireMessageCodec const wireCodec{};
	cucumber::internal::WireProtocolHandler protocolHandler{wireCodec, cukeEngine};

	std::unique_ptr<cucumber::internal::SocketServer> const socketServer =
		[&]() -> std::unique_ptr<cucumber::internal::SocketServer>
	{
		if (!unixPath.empty())
		{
			auto unixServer =
				std::make_unique<cucumber::internal::UnixSocketServer>(&protocolHandler);
			unixServer->listen(unixPath);
			if (verbose)
				std::clog << "Listening on socket " << unixServer->listenEndpoint() << std::endl;
			return unixServer;
		}
		else
		{
			auto tcpServer =
				std::make_unique<cucumber::internal::TCPSocketServer>(&protocolHandler);
			boost::asio::io_service service{};
			// Use resolver, rather than ip::from_string, in case "localhost" is given.
			tcpServer->listen(boost::asio::ip::tcp::resolver{service}
								  .resolve(host, std::to_string(port))
								  .begin()
								  ->endpoint());
			if (verbose)
				std::clog << "Listening on " << tcpServer->listenEndpoint() << std::endl;
			return tcpServer;
		}
	}();
};

/**
 * Synchronously open socket for connections, then asynchronously process them.
 *
 * This means when we run cucumber, there will be a server to connect to, even if the connections
 * are not yet being serviced, resolving the race condition.
 */
std::jthread start_wire_protocol_server(
	std::string host, int const port, std::string unixPath, bool const verbose)
{
	std::unique_ptr<WireServer> wire_server{
		new WireServer{std::move(host), port, std::move(unixPath), verbose}};

	return std::jthread([wire_server = std::move(wire_server)]
						{ wire_server->socketServer->acceptOnce(); });
}

/**
 * Launch a subprocess to execute `cucumber` command line.
 */
int run_cucumber(
	std::string const & feature_dir, std::string const & cucumber_options, bool const verbose)
{
	namespace bp = boost::process;

	bp::ipstream cucumber_stdout;
	std::string const cucumber_cmd = fmt::format(
		"{} {} {}", bp::search_path("cucumber").string(), cucumber_options, feature_dir);

	if (verbose)
		fmt::print("Executing '{}'\n", cucumber_cmd);

	bp::child cucumber = [&cucumber_cmd, &cucumber_stdout]
	{
		auto env = boost::this_process::environment();
		// Silence "THIS RUBY IMPLEMENTATION DOESN'T REPORT FILE AND LINE FOR PROCS"
		env["RUBY_IGNORE_CALLERS"] = "1";

		return bp::child{
			cucumber_cmd.c_str(),
			bp::std_in.close(),
			(bp::std_out & bp::std_err) > cucumber_stdout,
			env};
	}();

	int const exit_code = [&cucumber, &cucumber_cmd]
	{
		if (!cucumber.wait_for(std::chrono::milliseconds{10000}))
		{
			fmt::print(stderr, "Timeout executing '{}'\n", cucumber_cmd);
			cucumber.terminate();
			return 124;
		}
		return cucumber.exit_code();
	}();

	std::string const cucumber_output = [&cucumber_stdout]
	{
		std::stringstream ss;
		ss << cucumber_stdout.rdbuf();
		return ss.str();
	}();

	fmt::print("{}\n", cucumber_output);
	return exit_code;
}

}  // namespace

int main(int argc, char ** argv)
{
	// Ensure ctest, etc, output doesn't get interleaved.
	boost::scope_exit::aux::guard flusher{[] { std::cout.flush(); }};

	using boost::program_options::value;
	boost::program_options::options_description cmd_options_desc("Allowed options");

	cmd_options_desc.add_options()("help,h", "help for cucumber-cpp")(
		"verbose,v", "verbose output")(
		"config,c",
		value<std::string>()->default_value("step_definitions/cucumber.wire"),
		"location of .wire config file")(
		"features,f",
		value<std::string>()->default_value("."),
		"location of feature file or directory")(
		"options,o",
		value<std::string>()->default_value(""),
		"additional cucumber options (surround in quotes for multiple)");
	boost::program_options::variables_map cmd_options;
	boost::program_options::store(
		boost::program_options::parse_command_line(argc, argv, cmd_options_desc), cmd_options);
	boost::program_options::notify(cmd_options);

	if (cmd_options.count("help"))
	{
		cmd_options_desc.print(std::cerr, 80);
		return EXIT_SUCCESS;
	}

	if (!boost::filesystem::exists(cmd_options["config"].as<std::string>()))
	{
		fmt::print(
			stderr, "Wire config not found at '{}'", cmd_options["config"].as<std::string>());
		return EXIT_FAILURE;
	}

	std::string listenHost;
	int port{};
	std::string unixPath;

	std::string yaml;
	boost::filesystem::load_string_file(cmd_options["config"].as<std::string>(), yaml);
	auto config = YAML::Load(yaml);

	if (config["unix"].IsDefined())
	{
		unixPath = config["unix"].as<std::string>();
	}
	else if (config["host"].IsDefined())
	{
		listenHost = config["host"].as<std::string>();
		port = config["port"].as<int>();
	}

#ifndef BOOST_ASIO_HAS_LOCAL_SOCKETS
	if (!unixPath.empty())
	{
		fmt::print(stderr, "Unix paths are unsupported on this system: '{}'", unixPath);
		return EXIT_FAILURE;
	}
#endif

	bool const verbose = cmd_options.contains("verbose");

	try
	{
		[[maybe_unused]] auto server_thread =
			start_wire_protocol_server(listenHost, port, unixPath, verbose);

		return run_cucumber(
			cmd_options["features"].as<std::string>(),
			cmd_options["options"].as<std::string>(),
			verbose);
	}
	catch (std::exception & e)
	{
		fmt::print(stderr, "{}\n", e.what());
		return 1;
	}
}
