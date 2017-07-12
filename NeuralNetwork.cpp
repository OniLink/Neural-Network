#include "NeuralNetwork.hpp"

#include <cmath>

std::vector< float > NeuronLayer::propagate( std::vector< float > inputs ) {
	inputs.resize( input_count, 0.f ); // Ensure proper amount of input data.
	inputs[ input_count - 1 ] = 1.f; // Bias

	std::vector< float > outputs( output_count, 0.f );

	for( unsigned int y = 0; y < output_count; ++y ) {
		float output = 0.f;
		for( unsigned int x = 0; x < input_count; ++x ) {
			output += getWeight( x, y ) * inputs[ x ];
		}
		outputs[ y ] = output;
	}

	return outputs;
}

float NeuronLayer::getWeight( unsigned int input, unsigned int output ) const {
	input %= input_count;
	output %= output_count;
	return weights[ output * input_count + input ];
}

void NeuronLayer::setWeight( unsigned int input, unsigned int output, float weight ) {
	input %= input_count;
	output %= output_count;
	weights[ output * input_count + input ] = weight;
}

unsigned int NeuronLayer::getInputCount() const {
	return input_count - 1; // Minus 1 for bias input.
}

unsigned int NeuronLayer::getOutputCount() const {
	return output_count;
}

void NeuronLayer::setInputCount( unsigned int inputs ) {
	resizeMatrix( inputs, output_count );
}

void NeuronLayer::setOutputCount( unsigned int outputs ) {
	resizeMatrix( input_count, outputs );
}

void NeuronLayer::resizeMatrix( unsigned int new_inputs, unsigned int new_outputs ) {
	new_inputs += 1; // Add 1 for the bias input
	std::vector< float > new_weights( new_inputs * new_outputs, 0.f );

	for( unsigned int y = 0; y < output_count && y < new_outputs; ++y ) {
		for( unsigned int x = 0; x < input_count && x < new_inputs; ++x ) {
			new_weights[ y * new_inputs + x ] = getWeight( input, output );
		}
	}

	input_count = new_inputs;
	output_count = new_outputs;
	weights = new_weights;
}

std::vector< float > NeuralNetwork::propagate( std::vector< float > input_data ) {
	input_data.resize( input_layer.getInputCount() );

	std::vector< float > hidden_data = input_layer.propagate( input_data );
	hidden_data = flattenHiddenLayer( hidden_data );

	for( unsigned int i = 0; i < hidden_layers; ++i ) {
		hidden_data = hidden_layers[ i ].propagate( hidden_data );
		hidden_data = flattenHiddenLayer( hidden_data );
	}

	std::vector< float > output_data = output_layer.propagate( hidden_data );
	return flattenOutputLayer( output_data );
}

void NeuralNetwork::setInputCount( unsigned int input_count ) {
	input_layer.setInputCount( input_count );
}

void NeuralNetwork::setOutputCount( unsigned int output_count ) {
	output_layer.setOutputCount( output_count );
}

void NeuralNetwork::setHiddenNeuronCount( unsigned int hidden_neuron_count ) {
	hidden_neurons = hidden_neuron_count;
	input_layer.setOutputCount( hidden_neurons );
	output_layer.setInputCount( hidden_neurons );

	for( unsigned int i = 0; i < hidden_layers.size(); ++i ) {
		hidden_layers[ i ].setInputCount( hidden_neurons );
		hidden_layers[ i ].setOutputCount( hidden_neurons );
	}
}

const NeuronLayer& NeuralNetwork::getLayer( unsigned int layer_number ) const {
	return hidden_layers[ layer_number ];
}

void NeuralNetwork::setLayerCount( unsigned int hidden_layer_count ) {
	hidden_layers.resize( hidden_layer_count );
	setHiddenNeuronCount( hidden_neurons );
}

void NeuralNetwork::flattenHiddenLayer( const std::vector< float >& data ) {
	std::vector< float > flat( data.size() );

	for( unsigned int i = 0; i < data.size(); ++i ) {
		flat[ i ] = std::tanh( data[ i ] );
	}

	return flat;
}

void NeuralNetwork::flattenOutputLayer( const std::vector< float >& data ) {
	std::vector< float > flat( data.size() );

	for( unsigned int i = 0; i < data.size(); ++i ) {
		flat[ i ] = 1.f / ( 1.f + std::exp( -4.f * data[ i ] ) ); // Picked a rate of 4 so Taylor
																  // Expansion is 0.5 + x + O(x^3)
	}

	return flat;
}
