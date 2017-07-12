#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>

class NeuronLayer {
	public:
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

		/**
		 * Resize the weight matrix, preserving as many values as possible.
		 * Values are inserted/removed from the right and bottom.
		 * @param new_inputs The new number of inputs.
		 * @param new_outputs The new number of outputs.
		 */
		void resizeMatrix( unsigned int new_inputs, unsigned int new_outputs );
};

#endif // NEURAL_NETWORK_HPP
