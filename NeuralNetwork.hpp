#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>

class NeuronLayer {
	public:
		NeuronLayer();

		/**
		 * Propagate information through the layer.
		 * @param inputs The inputs to feed into the neuron layer.
		 * @return The output from the neuron layer.
		 */
		std::vector< float > propagate( std::vector< float > inputs );

		/**
		 * Get a weight from the weight matrix.
		 * @param input The input whose weight we want.
		 * @param output The output whose weight we want.
		 * @return The weight at (input, output).
		 */
		float getWeight( unsigned int input, unsigned int output ) const;

		/**
		 * Set a value in the weight matrix.
		 * @param input The input whose weight is affected.
		 * @param output The output whose weight is affected.
		 * @param weight The new weight for the input-output pair.
		 */
		void setWeight( unsigned int input, unsigned int output, float weight );

		/**
		 * Get the bias for a given output.
		 * @param output The output to get the bias for.
		 * @return The bias for the specified output.
		 */
		float getBias( unsigned int output );

		/**
		 * Set the bias for a given output.
		 * @param output The output to bias.
		 * @param new_bias The bias to give to the output.
		 */
		void setBias( unsigned int output, float new_bias );

		/**
		 * Get the number of inputs to the neural layer.
		 * @return The number of inputs to the neural layer.
		 */
		unsigned int getInputCount() const;

		/**
		 * Get the number of outputs from the neural layer.
		 * @return The number of outputs from the neural layer.
		 */
		unsigned int getOutputCount() const;

		/**
		 * Set the number of inputs to the neuron layer.
		 * @param inputs The number of inputs fed into the layer.
		 */
		void setInputCount( unsigned int inputs );

		/**
		 * Set the number of outputs from the neuron layer.
		 * @param outputs The number out outputs from the layer.
		 */
		void setOutputCount( unsigned int outputs );

	private:
		unsigned int input_count;
		unsigned int output_count;

		std::vector< float > weights;
		std::vector< float > bias;

		/**
		 * Resize the weight matrix, preserving as many values as possible.
		 * Values are inserted/removed from the right and bottom.
		 * @param new_inputs The new number of inputs.
		 * @param new_outputs The new number of outputs.
		 */
		void resizeMatrix( unsigned int new_inputs, unsigned int new_outputs );
};

class NeuralNetwork {
	public:
		NeuralNetwork();

		/**
		 * Propagate data through the neural network.
		 * @param input_data The data to feed in.
		 * @return The output of the neural network.
		 */
		std::vector< float > propagate( std::vector< float > input_data );

		/**
		 * Use back propagation to train the Neural Network.
		 * @param input_data The input data to train with.
		 * @param output_data The expected output data given the input.
		 * @param mutability How much the network is allowed to change.
		 */
		void backPropagate( std::vector< float > input_data, std::vector< float > output_data,
							 float mutability = 0.05 );

		/**
		 * Calculate the loss from the input to the output given an expected output.
		 * @param input_data The input to test for loss.
		 * @param output_data The expected output for the given input.
		 */
		float loss( std::vector< float > input_data, std::vector< float > output_data );

		/**
		 * Get the number of inputs to the network.
		 * @return The number of inputs to the network.
		 */
		unsigned int getInputCount() const;

		/**
		 * Get the number of outputs from the network.
		 * @return The number of outputs from the network.
		 */
		unsigned int getOutputCount() const;

		/**
		 * Set the number of inputs to the network.
		 * @param input_count The number of inputs to the network.
		 */
		void setInputCount( unsigned int input_count );

		/**
		 * Set the number of outputs from the network.
		 * @param output_count The number of outputs from the network.
		 */
		void setOutputCount( unsigned int output_count );

		/**
		 * Set the number of neurons found in each hidden layer.
		 * @param hidden_neuron_count The number of neurons per hidden layer.
		 */
		void setHiddenNeuronCount( unsigned int hidden_neuron_count );

		/**
		 * Get a hidden layer from the network.
		 * @param layer_number The layer to retrieve.
		 * @return The hidden layer at index (layer_number).
		 */
		NeuronLayer& getLayer( unsigned int layer_number );

		/**
		 * Set the number of hidden layers.
		 * @param hidden_layers The number of hidden layers.
		 */
		void setLayerCount( unsigned int hidden_layer_count );

	private:
		unsigned int hidden_neurons;

		NeuronLayer input_layer;
		NeuronLayer output_layer;

		std::vector< NeuronLayer > hidden_layers;
};

/**
 * Flatten data to the range [-1, 1].
 * @param data The data to flatten.
 * @return The flattened data.
 */
std::vector< float > flattenData( const std::vector< float >& data );

#endif // NEURAL_NETWORK_HPP
