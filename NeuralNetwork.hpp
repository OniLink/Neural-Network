#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <memory>
#include "json/json.h"
#include "NetworkLayer.hpp"
#include "FeedForwardLayer.hpp"
#include "LSTMLayer.hpp"

class NeuralNetwork {
	private:
		std::vector< std::shared_ptr< NetworkLayer > > m_layers;

	public:
		void loadFromJSON( Json::Value& layer_array ) {
			m_layers.clear();

			for( unsigned int i = 0; i < layer_array.size(); ++i ) {
				NetworkLayer* layer = nullptr;

				if( layer_array[ i ][ "type" ].asString() == std::string( "feedforward" ) || layer_array[ i ][ "type" ].asString() == std::string( "feed-forward" ) ) {
					layer = new FeedForwardLayer;
				} else if( layer_array[ i ][ "type" ].asString() == std::string( "lstm" ) ) {
					layer = new LSTMLayer;
				}

				if( layer == nullptr ) {
					continue;
				}

				layer->loadFromJSON( layer_array[ i ] );
				addLayer( layer );
			}
		}

		Json::Value saveToJSON() {
			Json::Value layer_array( Json::arrayValue );
			layer_array.resize( m_layers.size() );

			for( unsigned int i = 0; i < m_layers.size(); ++i ) {
				layer_array[ i ] = m_layers[ i ]->saveToJSON();
			}

			return layer_array;
		}

		/**
		 * Add a layer to the network. The network will take ownership of the layer and destroy it appropriately.
		 * @param layer The layer to add to the network. Number of inputs may be adjusted for compatability with the network.
		 */
		void addLayer( NetworkLayer* layer ) {
			if( !m_layers.empty() ) {
				layer->setInputCount( m_layers.back()->getOutputCount() );
			}

			m_layers.emplace_back( layer );
		}

		/**
		 * Propagate data through the neural network.
		 * @param input The data to propagate through the network.
		 * @return The output of the neural network.
		 */
		Vector propagate( Vector input ) {
			if( input.getDimension() != m_layers.front()->getInputCount() ) {
				throw std::string( "Invalid input size to network propagation" );
			}

			Vector data = input;

			for( auto& layer : m_layers ) {
				data = layer->propagate( data );
			}

			return data;
		}

		/**
		 * Train the neural network on some sample data.
		 * @param input The input to the neural network.
		 * @param output The expected output of the neural network.
		 * @param mutability The rate at which the network is allowed to adjust.
		 * @return The loss on the sample.
		 */
		float train( Vector input, Vector output, float mutability = 0.05f ) {
			if( input.getDimension() != m_layers.front()->getInputCount() ) {
				throw std::string( "Invalid input size to network training" );
			}

			if( output.getDimension() != m_layers.back()->getOutputCount() ) {
				throw std::string( "Invalid output size to network training" );
			}

			std::vector< Vector > results( m_layers.size() + 1 );
			results[ 0 ] = input;

			// Go forward to get the results
			for( unsigned int i = 0; i < m_layers.size(); ++i ) {
				results[ i + 1 ] = m_layers[ i ]->propagate( results[ i ] );
			}

			Vector delta;
			delta.setDimension( output.getDimension() );
			for( unsigned int i = 0; i < output.getDimension(); ++i ) {
				delta( i ) = results[ m_layers.size() ]( i ) - output( i );
			}

			// Go backwards to train
			for( int i = m_layers.size() - 1; i >= 0; --i ) {
				delta = m_layers[ i ]->train( results[ i ], results[ i + 1 ], delta, mutability );
			}

			float loss = 0.f;
			for( unsigned int i = 0; i < output.getDimension(); ++i ) {
				float error = output( i ) - results.back()( i );
				loss += 0.5f * error * error;
			}

			return loss;
		}

		void resetState() {
			for( unsigned int i = 0; i < m_layers.size(); ++i ) {
				m_layers[ i ]->resetState();
			}
		}
};

#endif // NEURALNETWORK_HPP
