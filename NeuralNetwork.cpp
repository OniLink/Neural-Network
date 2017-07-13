#include "NeuralNetwork.hpp"

#include <cmath>
#include <algorithm>

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

	return flattenData( outputs );
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
			new_weights[ y * new_inputs + x ] = getWeight( x, y );
		}
	}

	input_count = new_inputs;
	output_count = new_outputs;
	weights = new_weights;
}

std::vector< float > NeuralNetwork::propagate( std::vector< float > input_data ) {
	input_data.resize( input_layer.getInputCount() );

	std::vector< float > hidden_data = input_layer.propagate( input_data );

	for( unsigned int i = 0; i < hidden_layers.size(); ++i ) {
		hidden_data = hidden_layers[ i ].propagate( hidden_data );
	}

	return output_layer.propagate( hidden_data );
}

void NeuralNetwork::backPropagate( std::vector< float > input_data,
								   std::vector< float > output_data, float mutability ) {
	// Propagate forward and record results
	std::vector< float > input_outputs;
	std::vector< float > output_outputs;
	std::vector< std::vector< float > > intermediate_outputs;

	input_outputs = input_layer.propagate( input_data );
	std::vector< float > hidden_data = input_outputs;

	for( unsigned int i = 0; i < hidden_layers.size(); ++i ) {
		hidden_data = input_layer.propagate( hidden_data );
		intermediate_outputs.push_back( hidden_data );
	}

	output_outputs = output_layer.propagate( hidden_data );

	// Work backwards to calculate deltas
	std::vector< float > input_deltas;
	std::vector< float > output_deltas;
	std::vector< std::vector< float > > intermediate_deltas;

	for( unsigned int i = 0; i < output_outputs.size(); ++i ) {
		float neuron_value = output_outputs[ i ];
		float delta = ( neuron_value - output_data[ i ] ) * neuron_value * ( 1.f - neuron_value );
		output_deltas.push_back( delta );
	}

	intermediate_deltas.resize( hidden_layers.size() );

	for( int l = hidden_layers.size() - 1; l >= 0; --l ) {
		std::vector< float > deltas;
		NeuronLayer* previous_layer = &output_layer;
		std::vector< float > previous_deltas = output_deltas;
		if( l < hidden_layers.size() - 1 ) {
			previous_layer = &getLayer( l + 1 );
			previous_deltas = intermediate_deltas[ l + 1 ];
		}

		for( unsigned int i = 0; i < hidden_layers[ l ].getOutputCount(); ++i ) {
			float delta = 0.f;
			float neuron_value = intermediate_outputs[ l ][ i ];
			for( unsigned int k = 0; k < previous_layer->getOutputCount(); ++k ) {
				delta += previous_deltas[ k ] * previous_layer->getWeight( i, k );
			}
			delta *= neuron_value * ( 1.f - neuron_value );
			deltas.push_back( delta );
		}
		intermediate_deltas[ l ] = deltas;
	}

	NeuronLayer* previous_layer = &output_layer;
	std::vector< float > previous_deltas = output_deltas;
	if( hidden_layers.size() > 0 ) {
		previous_layer = &getLayer( 0 );
		previous_deltas = intermediate_deltas[ 0 ];
	}

	for( unsigned int i = 0; i < input_layer.getOutputCount(); ++i ) {
		float delta = 0.f;
		float neuron_value = input_outputs[ i ];
		for( unsigned int k = 0; k < previous_layer->getOutputCount(); ++k ) {
			delta += previous_deltas[ k ] * previous_layer->getWeight( i, k );
		}
		delta *= neuron_value * ( 1.f - neuron_value );
		input_deltas.push_back( delta );
	}

	// Update weights
	for( unsigned int i = 0; i < input_layer.getOutputCount(); ++i ) {
		for( unsigned int j = 0; j < input_data.size(); ++j ) {
			float adjust = mutability * input_deltas[ i ] * input_data[ j ];
			input_layer.setWeight( j, i, input_layer.getWeight( j, i ) - adjust );
		}
	}

	for( unsigned int k = 0; k < hidden_layers.size(); ++k ) {
		for( unsigned int i = 0; i < getLayer( k ).getOutputCount(); ++i ) {
			for( unsigned int j = 0; j < getLayer( k ).getInputCount(); ++j ) {
				float adjust = mutability * intermediate_deltas[ k ][ i ];

				if( k == 0 ) {
					adjust *= input_outputs[ j ];
				} else {
					adjust *= intermediate_outputs[ k - 1 ][ j ];
				}

				getLayer( k ).setWeight( j, i, getLayer( k ).getWeight( j, i ) - adjust );
			}
		}
	}

	for( unsigned int i = 0; i < output_layer.getOutputCount(); ++i ) {
		for( unsigned int j = 0; j < output_layer.getInputCount(); ++j ) {
			float adjust = mutability * output_deltas[ i ];

			if( hidden_layers.size() == 0 ) {
				adjust *= intermediate_outputs.back()[ j ];
			} else {
				adjust *= input_outputs[ j ];
			}

			output_layer.setWeight( j, i, output_layer.getWeight( j, i ) - adjust );
		}
	}
}

float NeuralNetwork::loss( std::vector< float > input_data, std::vector< float > output_data ) {
	input_data.resize( input_layer.getInputCount() );
	output_data.resize( output_layer.getOutputCount() );
	std::vector< float > actual_output = propagate( input_data );

	float loss_value = 0.f;

	for( unsigned int i = 0; i < actual_output.size(); ++i ) {
		float error = output_data[ i ] - actual_output[ i ];
		loss_value += error * error;
	}

	return 0.5f * loss_value;
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

NeuronLayer& NeuralNetwork::getLayer( unsigned int layer_number ) {
	return hidden_layers[ layer_number ];
}

void NeuralNetwork::setLayerCount( unsigned int hidden_layer_count ) {
	hidden_layers.resize( hidden_layer_count );
	setHiddenNeuronCount( hidden_neurons );
}

std::vector< float > flattenData( const std::vector< float >& data ) {
	std::vector< float > flat( data.size() );

	for( unsigned int i = 0; i < data.size(); ++i ) {
		flat[ i ] = 1.f / ( 1.f + std::exp( -data[ i ] ) );
	}

	return flat;
}
