#include "NeuralNetwork.hpp"

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
