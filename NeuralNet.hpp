#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <algorithm>
#include <cmath>
#include "LinearAlgebra.hpp"

/**
 * Apply the logistic flattening function to a data vector.
 * @param input The data to apply the logistic map to.
 * @return The flattened data.
 */
Vector< float > logistic( Vector< float > input ) {
	for( unsigned int i = 0; i < input.getLength(); ++i ) {
		input.at( i ) = 1.f / ( 1.f + std::exp( -input.at( i ) ) );
	}
	return input;
}

Vector< float > logisticDerivative( Vector< float > input ) {
	for( unsigned int i = 0; i < input.getLength(); ++i ) {
		input.at( i ) = input.at( i ) * ( 1.f - input.at( i ) );
	}
	return input;
}

class NeuralNetwork {
	private:
		unsigned int inputs;
		unsigned int outputs;
		unsigned int layers;
		unsigned int neurons;

		Matrix< float > input_layer;
		Vector< float > input_bias;

		std::vector< Matrix< float > > hidden_layers;
		std::vector< Vector< float > > hidden_biases;

		Matrix< float > output_layer;
		Vector< float > output_bias;

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
			input_layer( input_count, neuron_count ), input_bias( neuron_count ),
			output_layer( neuron_count, output_count ), output_bias( output_count ) {
			for( unsigned int i = 0; i < layers; ++i ) {
				hidden_layers.emplace_back( neuron_count, neuron_count );
				hidden_biases.emplace_back( neuron_count );
			}
		}

		/**
		 * Propagate data through the neural network.
		 * @param input The input to run through the neural network.
		 * @return The output data from the neural network.
		 */
		Vector< float > propagate( Vector< float >& input ) {
			if( input.getLength() != inputs ) {
				return Vector< float >( outputs );
			}

			Vector< float > hidden_data = logistic( input_layer * input + input_bias );

			for( unsigned int i = 0; i < hidden_layers.size(); ++i ) {
				hidden_data = logistic( hidden_layers[ i ] * hidden_data + hidden_biases[ i ] );
			}

			return logistic( output_layer * hidden_data + output_bias );
		}

		/**
		 * Use back propagation to train the network.
		 * @param input The input data to train on.
		 * @param output The expected output data.
		 * @param mutability The amount by which the neural network is allowed to change.
		 */
		void backPropagate( Vector< float >& input, Vector< float >& output, float mutability = 0.05f ) {
			// Propagate forward and record results
			Vector< float > input_results = logistic( input_layer * input + input_bias );
			Vector< float > hidden_results = input_results;
			std::vector< Vector< float > > intermediate_results( hidden_layers.size(), neurons );
			for( unsigned int i = 0; i < hidden_layers.size(); ++i ) {
				hidden_results = logistic( hidden_layers[ i ] * hidden_results + hidden_biases[ i ] );
				intermediate_results[ i ] =  hidden_results;
			}
			Vector< float > output_results = logistic( output_layer * hidden_results + output_bias );

			// Work backwards to calculate deltas
			Vector< float > output_deltas = ( output_results - output ).hadamard( logisticDerivative( output_results ) );
			Vector< float > previous_deltas = output_deltas;
			Matrix< float > previous_layer = output_layer;
			std::vector< Vector< float > > intermediate_deltas( hidden_layers.size(), neurons );

			for( int i = hidden_layers.size() - 1; i >= 0; --i ) {
				Vector< float > delta_left = previous_deltas * previous_layer;
				Vector< float > delta_right = logisticDerivative( intermediate_results[ i ] );
				Vector< float > deltas = delta_left.hadamard( delta_right );
				intermediate_deltas[ i ] = deltas;

				previous_layer = hidden_layers[ i ];
				previous_deltas = deltas;
			}

			Vector< float > input_deltas = ( previous_deltas * previous_layer ).hadamard( logisticDerivative( input_results ) );

			// Update Weights
			input_layer = input_layer - mutability * input * input_deltas;
			input_bias = input_bias - mutability * input_deltas;
			Vector< float > previous_results = input_results;

			for( unsigned int i = 0; i < hidden_layers.size(); ++i ) {
				hidden_layers[ i ] = hidden_layers[ i ] - mutability * previous_results * intermediate_deltas[ i ];
				hidden_biases[ i ] = hidden_biases[ i ] - mutability * intermediate_deltas[ i ];
				previous_results = intermediate_results[ i ];
			}

			output_layer = output_layer - mutability * previous_results * output_deltas;
			output_bias = output_bias - mutability * output_deltas;
		}

		float loss( Vector< float >& input, Vector< float >& output ) {
			if( input.getLength() != inputs || output.getLength() != outputs ) {
				return -1.f;
			}

			Vector< float > results = propagate( input );
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
