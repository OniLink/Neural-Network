#ifndef FEEDFORWARDLAYER_HPP
#define FEEDFORWARDLAYER_HPP

#include <cmath>
#include <random>
#include "Matrix.hpp"
#include "NetworkLayer.hpp"
#include "Vector.hpp"

class FeedForwardLayer : public NetworkLayer {
	private:
		Matrix m_weights;
		Vector m_bias;

		float activation( float input ) {
			//return 1.f / ( 1.f + std::exp( -input ) );
			return std::tanh( input );
		}

		float activationOutputDerivative( float output ) {
			//return output * ( 1.f - output );
			return ( 1.f - output * output );
		}

		float activationDerivative( float input ) {
			return activationOutputDerivative( activation( input ) );
		}

	protected:
		virtual void setSizeInternal( const unsigned int inputs, const unsigned int outputs ) {
			m_weights.setSize( outputs, inputs );
			m_bias.setDimension( outputs );

			// Init randomly to make rows unique
			std::random_device device;
			std::mt19937 generator( device() );
			std::uniform_real_distribution<> distribution( -1.f, 1.f );
			for( unsigned int x = 0; x < inputs; ++x ) {
				for( unsigned int y = 0; y < outputs; ++y ) {
					m_weights( y, x ) = distribution( generator );
				}
			}

			for( unsigned int y = 0; y < outputs; ++y ) {
				m_bias( y ) = distribution( generator );
			}
		}

		virtual void loadFromJSONInternal( Json::Value& data_value ) {
			Json::Value weights = data_value[ "weights" ];
			Json::Value bias = data_value[ "bias" ];

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					m_weights( y, x ) = weights[ y * getInputCount() + x ].asFloat();
				}

				m_bias( y ) = bias[ y ].asFloat();
			}
		}

		virtual Json::Value saveToJSONInternal() {
			Json::Value data_object( Json::objectValue );

			Json::Value weights( Json::arrayValue );
			weights.resize( getInputCount() * getOutputCount() );

			Json::Value bias( Json::arrayValue );
			bias.resize( getOutputCount() );

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					weights[ y * getInputCount() + x ] = m_weights( y, x );
				}

				bias[ y ] = m_bias( y );
			}

			data_object[ "weights" ] = weights;
			data_object[ "bias" ] = bias;

			return data_object;
		}

		virtual std::string getJSONTypeName() const {
			return std::string( "feed-forward" );
		}

	public:
		virtual Vector propagate( Vector input ) {
			if( input.getDimension() != getInputCount() ) {
				throw std::string( "Invalid input size to layer propagation" );
			}

			Vector output;
			output.setDimension( getOutputCount() );

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				float accum = m_bias( y );

				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					accum += m_weights( y, x ) * input( x );
				}

				output( y ) = activation( accum );
			}

			return output;
		}

		virtual Vector train( Vector input, Vector output, Vector delta, float mutability = 0.05f ) {
			if( input.getDimension() != getInputCount() ) {
				throw std::string( "Invalid input size to layer training" );
			}

			if( delta.getDimension() != getOutputCount() ) {
				throw std::string( "Invalid delta size to layer training" );
			}

			if( output.getDimension() != getOutputCount() ) {
				throw std::string( "Invalid output size to layer training" );
			}

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				delta( y ) *= activationOutputDerivative( output( y ) );
			}

			Vector new_delta;
			new_delta.setDimension( getInputCount() );

			for( unsigned int x = 0; x < getInputCount(); ++x ) {
				float accum = 0.f;

				for( unsigned int y = 0; y < getOutputCount(); ++x ) {
					accum += delta( y ) * m_weights( y, x );
				}

				new_delta( x ) = accum;
			}

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					m_weights( y, x ) -= mutability * delta( y ) * input( x );
				}

				m_bias( y ) -= mutability * delta( y );
			}

			return new_delta;
		}
};

#endif // FEEDFORWARDLAYER_HPP
