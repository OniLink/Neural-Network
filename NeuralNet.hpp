#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include "cl.hpp"
#include "LinearAlgebra.hpp"

const std::string code_propagate_layer =
	"__kernel void propagate_layer( const unsigned int N, __global const float* input,\n"
	"								__global const float* matrix, __global const float* bias,\n"
	"								__global float* output ) {\n"
	"	const int index = get_global_id( 0 );\n"
	"	float accum = bias[ index ];\n"
	"	for( int i = 0; i < N; ++i ) {\n"
	"		accum += matrix[ index * N + i ] * input[ i ];\n"
	"	}\n"
	"	output[ index ] = 1.f / ( 1.f + exp( -accum ) )\n;"
	"}\n";

/**
 * Apply the logistic flattening function to a data vector.
 * @param input The data to apply the logistic map to.
 * @return The flattened data.
 */
Vector logistic( Vector input ) {
	for( unsigned int i = 0; i < input.getLength(); ++i ) {
		input.at( i ) = 1.f / ( 1.f + std::exp( -input.at( i ) ) );
	}
	return input;
}

Vector logisticDerivative( Vector input ) {
	for( unsigned int i = 0; i < input.getLength(); ++i ) {
		input.at( i ) = input.at( i ) * ( 1.f - input.at( i ) );
	}
	return input;
}

class NeuralNetwork {
	private:
		bool use_opencl;
		cl::Context opencl_context;
		cl::CommandQueue opencl_queue;

		cl::Kernel kernel_propagate_layer;

		unsigned int inputs;
		unsigned int outputs;
		unsigned int layers;
		unsigned int neurons;

		Matrix input_layer;
		Vector input_bias;

		std::vector< Matrix > hidden_layers;
		std::vector< Vector > hidden_biases;

		Matrix output_layer;
		Vector output_bias;

		bool initializeOpenCL() {
			// Set up the OpenCL Context
			std::vector< cl::Platform > platforms;
			cl::Platform::get( &platforms );
			if( platforms.size() == 0 ) {
				use_opencl = false;
				std::cout << "OpenCL Initialization Error: No platforms found\n";
				return false;
			}
			cl::Platform platform = platforms[ 0 ];

			std::vector< cl::Device > devices;
			platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );
			if( devices.size() == 0 ) {
				use_opencl = false;
				std::cout << "OpenCL Initialization Error: No devices found\n";
				return false;
			}
			cl::Device device = devices[ 0 ];

			opencl_context = cl::Context( { device } );

			// Set up the neural network program
			cl::Program::Sources program_sources;
			program_sources.push_back( { code_propagate_layer.data(), code_propagate_layer.length() } );

			cl::Program program( opencl_context, program_sources );
			if( program.build( { device } ) != CL_SUCCESS ) {
				use_opencl = false;
				std::cout << "OpenCL Initialization Error: Failed to build OpenCL program\n";
				std::cout << "Build Log: " << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device ) << std::endl;
				return false;
			}

			kernel_propagate_layer = cl::Kernel( program, "propagate_layer" );

			// Set up the command queue
			opencl_queue = cl::CommandQueue( opencl_context, device );

			// Done
			return true;
		}

		Vector propagateLayerCPU( Vector& input, Matrix& matrix, Vector& bias ) {
			return logistic( matrix * input + bias );
		}

		Vector propagateLayerCL( Vector& input, Matrix& matrix, Vector& bias ) {
			Vector output( matrix.getHeight() );
			unsigned int matrix_width = matrix.getWidth();

			cl::Buffer width_buffer( opencl_context, CL_MEM_READ_ONLY,
									 sizeof( unsigned int ) );
			cl::Buffer input_buffer( opencl_context, CL_MEM_READ_ONLY,
									 sizeof( float ) * input.getLength() );
			cl::Buffer matrix_buffer( opencl_context, CL_MEM_READ_ONLY,
									  sizeof( float ) * matrix.getWidth() * matrix.getHeight() );
			cl::Buffer bias_buffer( opencl_context, CL_MEM_READ_ONLY,
									sizeof( float ) * bias.getLength() );
			cl::Buffer output_buffer( opencl_context, CL_MEM_WRITE_ONLY,
									  sizeof( float ) * matrix.getHeight() );

			opencl_queue.enqueueWriteBuffer( width_buffer, CL_FALSE, 0,
											 sizeof( unsigned int ),
											 &matrix_width );
			opencl_queue.enqueueWriteBuffer( input_buffer, CL_FALSE, 0,
											 sizeof( float ) * input.getLength(),
											 input.getInternalData().data() );
			opencl_queue.enqueueWriteBuffer( matrix_buffer, CL_FALSE, 0,
											 sizeof( float ) * matrix.getWidth() * matrix.getHeight(),
											 input.getInternalData().data() );
			opencl_queue.enqueueWriteBuffer( bias_buffer, CL_FALSE, 0,
											 sizeof( float ) * bias.getLength(),
											 bias.getInternalData().data() );

			kernel_propagate_layer.setArg( 0, width_buffer );
			kernel_propagate_layer.setArg( 1, input_buffer );
			kernel_propagate_layer.setArg( 2, matrix_buffer );
			kernel_propagate_layer.setArg( 3, bias_buffer );
			kernel_propagate_layer.setArg( 4, output_buffer );

			opencl_queue.enqueueNDRangeKernel( kernel_propagate_layer, cl::NullRange,
											   cl::NDRange( matrix.getHeight() ), cl::NullRange );
			opencl_queue.finish();
			opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * matrix.getHeight(),
											output.getInternalData().data() );

			return output;
		}

	public:
		/**
		 * Create a neural network.
		 * @param input_count The number of inputs to the neural network.
		 * @param output_count The number of outputs from the neural network.
		 * @param layer_count The number of hidden layers in the neural network.
		 * @param neuron_count The number of neurons in each hidden layer.
		 */
		NeuralNetwork( unsigned int input_count, unsigned int output_count,
					   unsigned int layer_count, unsigned int neuron_count ) :
			inputs( input_count ), outputs( output_count ),
			layers( layer_count ), neurons( neuron_count ),
			input_layer( neuron_count, input_count ), input_bias( neuron_count ),
			output_layer( output_count, neuron_count ), output_bias( output_count ) {
			for( unsigned int i = 0; i < layers; ++i ) {
				hidden_layers.emplace_back( neuron_count, neuron_count );
				hidden_biases.emplace_back( neuron_count );
			}

			use_opencl = initializeOpenCL();
		}

		Vector propagateLayer( Vector& input, Matrix& matrix, Vector& bias ) {
			if( input.getLength() != matrix.getWidth() || matrix.getHeight() != bias.getLength() ) {
				std::cout << "Cannot propagate layer:\n";
				std::cout << "Input length: " << input.getLength() << std::endl;
				std::cout << "Matrix width: " << matrix.getWidth() << std::endl;
				std::cout << "Matrix height: " << matrix.getHeight() << std::endl;
				std::cout << "Bias length: " << bias.getLength() << std::endl;
				return Vector( 0 );
			}

			Vector output( matrix.getHeight() );

			if( !use_opencl ) {
				output = propagateLayerCPU( input, matrix, bias );
			} else {
				output = propagateLayerCL( input, matrix, bias );
			}

			return output;
		}

		/**
		 * Propagate data through the neural network.
		 * @param input The input to run through the neural network.
		 * @return The output data from the neural network.
		 */
		Vector propagate( Vector& input ) {
			if( input.getLength() != inputs ) {
				return Vector( outputs );
			}

			Vector hidden_data = propagateLayer( input, input_layer, input_bias );

			for( unsigned int i = 0; i < hidden_layers.size(); ++i ) {
				hidden_data = propagateLayer( hidden_data, hidden_layers[ i ], hidden_biases[ i ] );
			}

			return propagateLayer( hidden_data, output_layer, output_bias );
		}

		/**
		 * Use back propagation to train the network.
		 * @param input The input data to train on.
		 * @param output The expected output data.
		 * @param mutability The amount by which the neural network is allowed to change.
		 */
		void backPropagate( Vector& input, Vector& output, float mutability = 0.05f ) {
			// Propagate forward and record results
			Vector input_results = propagateLayer( input, input_layer, input_bias );
			Vector hidden_results = input_results;

			std::vector< Vector > intermediate_results( hidden_layers.size(), neurons );
			for( unsigned int i = 0; i < hidden_layers.size(); ++i ) {
				hidden_results = propagateLayer( hidden_results, hidden_layers[ i ], hidden_biases[ i ] );
				intermediate_results[ i ] =  hidden_results;
			}

			Vector output_results = propagateLayer( hidden_results, output_layer, output_bias );

			// Work backwards to calculate deltas
			Vector output_deltas = ( output_results - output ).hadamard( logisticDerivative( output_results ) );
			Vector previous_deltas = output_deltas;
			Matrix previous_layer = output_layer;
			std::vector< Vector > intermediate_deltas( hidden_layers.size(), neurons );

			for( int i = hidden_layers.size() - 1; i >= 0; --i ) {
				Vector delta_left = previous_deltas * previous_layer;
				Vector delta_right = logisticDerivative( intermediate_results[ i ] );
				Vector deltas = delta_left.hadamard( delta_right );
				intermediate_deltas[ i ] = deltas;

				previous_layer = hidden_layers[ i ];
				previous_deltas = deltas;
			}

			Vector input_deltas = ( previous_deltas * previous_layer ).hadamard( logisticDerivative( input_results ) );

			// Update Weights
			input_layer = input_layer - mutability * input * input_deltas;
			input_bias = input_bias - mutability * input_deltas;
			Vector previous_results = input_results;

			for( unsigned int i = 0; i < hidden_layers.size(); ++i ) {
				hidden_layers[ i ] = hidden_layers[ i ] - mutability * previous_results * intermediate_deltas[ i ];
				hidden_biases[ i ] = hidden_biases[ i ] - mutability * intermediate_deltas[ i ];
				previous_results = intermediate_results[ i ];
			}

			output_layer = output_layer - mutability * previous_results * output_deltas;
			output_bias = output_bias - mutability * output_deltas;
		}

		float loss( Vector& input, Vector& output ) {
			if( input.getLength() != inputs || output.getLength() != outputs ) {
				return -1.f;
			}

			Vector results = propagate( input );
			float loss_value = 0.f;
			for( unsigned int i = 0; i < output.getLength(); ++i ) {
				float error = results.at( i ) - output.at( i );
				loss_value += error * error;
			}
			loss_value *= 0.5f;

			return loss_value;
		}
};

#endif
