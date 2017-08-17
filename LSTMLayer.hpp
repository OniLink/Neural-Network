#ifndef LSTMLAYER_HPP
#define LSTMLAYER_HPP

#include <cmath>
#include <vector>
#include "Matrix.hpp"
#include "NetworkLayer.hpp"
#include "Vector.hpp"

class LSTMLayer : public NetworkLayer {
	private:
		Matrix m_forget_weights;
		Matrix m_learn_weights;
		Matrix m_cell_weights;
		Matrix m_output_weights;

		Matrix m_forget_state_weights;
		Matrix m_learn_state_weights;
		Matrix m_cell_state_weights;
		Matrix m_output_state_weights;

		Vector m_forget_bias;
		Vector m_learn_bias;
		Vector m_cell_bias;
		Vector m_output_bias;

		Vector m_cell_state;
		Vector m_previous_output;
		Vector m_train_state;
		Vector m_train_output;

		float activation( float input ) {
			return 1.f / ( 1.f + std::exp( -input ) );
		}

		float activationOutputDerivative( float output ) {
			return output * ( 1.f - output );
		}

		float activationDerivative( float input ) {
			return activationOutputDerivative( activation( input ) );
		}

		float cellActivation( float input ) {
			return std::tanh( input );
		}

		float cellActivationOutputDerivative( float output ) {
			return 1.f - output * output;
		}

		float cellActivationDerivative( float input ) {
			return cellActivationOutputDerivative( cellActivation( input ) );
		}

		Vector calculateForgetVector( Vector input, Vector previous_output ) {
			Vector result;
			result.setDimension( getOutputCount() );

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				float accum = m_forget_bias( y );

				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					accum += m_forget_weights( y, x ) * input( x );
				}

				for( unsigned int x = 0; x < getOutputCount(); ++x ) {
					accum += m_forget_state_weights( y, x ) * previous_output( x );
				}

				result( y ) = activation( accum );
			}

			return result;
		}

		Vector calculateLearnVector( Vector input, Vector previous_output ) {
			Vector result;
			result.setDimension( getOutputCount() );

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				float accum = m_learn_bias( y );

				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					accum += m_learn_weights( y, x ) * input( x );
				}

				for( unsigned int x = 0; x < getOutputCount(); ++x ) {
					accum += m_learn_state_weights( y, x ) * previous_output( x );
				}

				result( y ) = activation( accum );
			}

			return result;
		}

		Vector calculateInformationVector( Vector input, Vector previous_output ) {
			Vector result;
			result.setDimension( getOutputCount() );

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				float accum = m_cell_bias( y );

				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					accum += m_cell_weights( y, x ) * input( x );
				}

				for( unsigned int x = 0; x < getOutputCount(); ++x ) {
					accum += m_cell_state_weights( y, x ) * previous_output( x );
				}

				result( y ) = cellActivation( accum );
			}

			return result;
		}

		void updateCellState( Vector input ) {
			Vector forget_vector = calculateForgetVector( input, m_previous_output );
			Vector learn_vector = calculateLearnVector( input, m_previous_output );
			Vector information_vector = calculateInformationVector( input, m_previous_output );

			m_train_state = m_cell_state;

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {

				m_cell_state( y ) = forget_vector( y ) * m_train_state( y ) + learn_vector( y ) * information_vector( y );
			}
		}

		Vector calculateOutputVector( Vector input, Vector previous_output ) {
			Vector result;
			result.setDimension( getOutputCount() );

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				result( y ) = m_output_bias( y );

				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					result( y ) += m_output_weights( y, x ) * input( x );
				}

				for( unsigned int x = 0; x < getOutputCount(); ++x ) {
					result( y ) += m_output_state_weights( y, x ) * previous_output( x );
				}

				result( y ) = activation( result( y ) );
			}

			return result;
		}

	protected:
		virtual void setSizeInternal( const unsigned int inputs, const unsigned int outputs ) {
			m_forget_weights.setSize( outputs, inputs );
			m_learn_weights.setSize( outputs, inputs );
			m_cell_weights.setSize( outputs, inputs );
			m_output_weights.setSize( outputs, inputs );

			m_forget_state_weights.setSize( outputs, outputs );
			m_learn_state_weights.setSize( outputs, outputs );
			m_cell_state_weights.setSize( outputs, outputs );
			m_output_state_weights.setSize( outputs, outputs );

			m_forget_bias.setDimension( outputs );
			m_learn_bias.setDimension( outputs );
			m_cell_bias.setDimension( outputs );
			m_output_bias.setDimension( outputs );

			m_cell_state.setDimension( outputs );
			m_previous_output.setDimension( outputs );
			m_train_state.setDimension( outputs );
			m_train_output.setDimension( outputs );

			// Init randomly to make rows unique
			std::random_device device;
			std::mt19937 generator( device() );
			std::uniform_real_distribution<> distribution( -0.01f, 0.01f );

			for( unsigned int x = 0; x < inputs; ++x ) {
				for( unsigned int y = 0; y < outputs; ++y ) {
					m_forget_weights( y, x ) = distribution( generator );
					m_learn_weights( y, x ) = distribution( generator );
					m_cell_weights( y, x ) = distribution( generator );
					m_output_weights( y, x ) = distribution( generator );
				}
			}

			for( unsigned int x = 0; x < outputs; ++x ) {
				for( unsigned int y = 0; y < outputs; ++y ) {
					m_forget_state_weights( y, x ) = distribution( generator );
					m_learn_state_weights( y, x ) = distribution( generator );
					m_cell_state_weights( y, x ) = distribution( generator );
					m_output_state_weights( y, x ) = distribution( generator );
				}
			}

			for( unsigned int y = 0; y < outputs; ++y ) {
				m_forget_bias( y ) = distribution( generator );
				m_learn_bias( y ) = distribution( generator );
				m_cell_bias( y ) = distribution( generator );
				m_output_bias( y ) = distribution( generator );
			}
		}

		virtual void loadFromJSONInternal( Json::Value& data_value ) {
			Json::Value forget_weights = data_value[ "forget-weights" ];
			Json::Value learn_weights = data_value[ "learn-weights" ];
			Json::Value cell_weights = data_value[ "cell-weights" ];
			Json::Value output_weights = data_value[ "output-weights" ];

			Json::Value forget_state_weights = data_value[ "forget-state-weights" ];
			Json::Value learn_state_weights = data_value[ "learn-state-weights" ];
			Json::Value cell_state_weights = data_value[ "cell-state-weights" ];
			Json::Value output_state_weights = data_value[ "output-state-weights" ];

			Json::Value forget_bias = data_value[ "forget-bias" ];
			Json::Value learn_bias = data_value[ "learn-bias" ];
			Json::Value cell_bias = data_value[ "cell-bias" ];
			Json::Value output_bias = data_value[ "output-bias" ];

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					m_forget_weights( y, x ) = forget_weights[ y * getInputCount() + x ].asFloat();
					m_learn_weights( y, x ) = learn_weights[ y * getInputCount() + x ].asFloat();
					m_cell_weights( y, x ) = cell_weights[ y * getInputCount() + x ].asFloat();
					m_output_weights( y, x ) = output_weights[ y * getInputCount() + x ].asFloat();
				}

				m_forget_bias( y ) = forget_bias[ y ].asFloat();
				m_learn_bias( y ) = learn_bias[ y ].asFloat();
				m_cell_bias( y ) = cell_bias[ y ].asFloat();
				m_output_bias( y ) = output_bias[ y ].asFloat();
			}

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				for( unsigned int x = 0; x < getOutputCount(); ++x ) {
					m_forget_state_weights( y, x ) = forget_state_weights[ y * getOutputCount() + x ].asFloat();
					m_learn_state_weights( y, x ) = learn_state_weights[ y * getOutputCount() + x ].asFloat();
					m_cell_state_weights( y, x ) = cell_state_weights[ y * getOutputCount() + x ].asFloat();
					m_output_state_weights( y, x ) = output_state_weights[ y * getOutputCount() + x ].asFloat();
				}
			}
		}

		virtual Json::Value saveToJSONInternal() {
			Json::Value data_object( Json::objectValue );

			Json::Value forget_weights( Json::arrayValue );
			Json::Value learn_weights( Json::arrayValue );
			Json::Value cell_weights( Json::arrayValue );
			Json::Value output_weights( Json::arrayValue );
			forget_weights.resize( getInputCount() * getOutputCount() );
			learn_weights.resize( getInputCount() * getOutputCount() );
			cell_weights.resize( getInputCount() * getOutputCount() );
			output_weights.resize( getInputCount() * getOutputCount() );

			Json::Value forget_state_weights( Json::arrayValue );
			Json::Value learn_state_weights( Json::arrayValue );
			Json::Value cell_state_weights( Json::arrayValue );
			Json::Value output_state_weights( Json::arrayValue );
			forget_state_weights.resize( getOutputCount() * getOutputCount() );
			learn_state_weights.resize( getOutputCount() * getOutputCount() );
			cell_state_weights.resize( getOutputCount() * getOutputCount() );
			output_state_weights.resize( getOutputCount() * getOutputCount() );

			Json::Value forget_bias( Json::arrayValue );
			Json::Value learn_bias( Json::arrayValue );
			Json::Value cell_bias( Json::arrayValue );
			Json::Value output_bias( Json::arrayValue );
			forget_bias.resize( getOutputCount() );
			learn_bias.resize( getOutputCount() );
			cell_bias.resize( getOutputCount() );
			output_bias.resize( getOutputCount() );

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					forget_weights[ y * getInputCount() + x ] = m_forget_weights( y, x );
					learn_weights[ y * getInputCount() + x ] = m_learn_weights( y, x );
					cell_weights[ y * getInputCount() + x ] = m_cell_weights( y, x );
					output_weights[ y * getInputCount() + x ] = m_output_weights( y, x );
				}

				forget_bias[ y ] = m_forget_bias( y );
				learn_bias[ y ] = m_learn_bias( y );
				cell_bias[ y ] = m_cell_bias( y );
				output_bias[ y ] = m_output_bias( y );
			}

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				for( unsigned int x = 0; x < getOutputCount(); ++x ) {
					forget_state_weights[ y * getOutputCount() + x ] = m_forget_state_weights( y, x );
					learn_state_weights[ y * getOutputCount() + x ] = m_learn_state_weights( y, x );
					cell_state_weights[ y * getOutputCount() + x ] = m_cell_state_weights( y, x );
					output_state_weights[ y * getOutputCount() + x ] = m_output_state_weights( y, x );
				}
			}

			data_object[ "forget-weights" ] = forget_weights;
			data_object[ "learn-weights" ] = learn_weights;
			data_object[ "cell-weights" ] = cell_weights;
			data_object[ "output-weights" ] = output_weights;

			data_object[ "forget-state-weights" ] = forget_state_weights;
			data_object[ "learn-state-weights" ] = learn_state_weights;
			data_object[ "cell-state-weights" ] = cell_state_weights;
			data_object[ "output-state-weights" ] = output_state_weights;

			data_object[ "forget-bias" ] = forget_bias;
			data_object[ "learn-bias" ] = learn_bias;
			data_object[ "cell-bias" ] = cell_bias;
			data_object[ "output-bias" ] = output_bias;

			return data_object;
		}

		virtual std::string getJSONTypeName() const {
			return std::string( "lstm" );
		}

	public:
		virtual Vector propagate( Vector input ) {
			if( input.getDimension() != getInputCount() ) {
				throw std::string( "Invalid input size to layer propagation" );
			}

			Vector output_vector = calculateOutputVector( input, m_previous_output );
			updateCellState( input );

			Vector output;
			output.setDimension( getOutputCount() );

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				output( y ) = cellActivation( output_vector( y ) * m_cell_state( y ) );
			}

			m_train_output = m_previous_output;
			m_previous_output = output;

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
				delta( y ) *= cellActivationOutputDerivative( output( y ) );
			}

			Vector forget_vector = calculateForgetVector( input, m_train_output );
			Vector learn_vector = calculateLearnVector( input, m_train_output );
			Vector information_vector = calculateInformationVector( input, m_train_output );
			Vector output_vector = calculateOutputVector( input, m_train_output );

			Vector forget_delta = delta;
			Vector learn_delta = delta;
			Vector cell_delta = delta;
			Vector output_delta = delta;

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				forget_delta( y ) *= output_vector( y ) * m_train_state( y ) * activationOutputDerivative( forget_vector( y ) );
				learn_delta( y ) *= output_vector( y ) * information_vector( y ) * activationOutputDerivative( learn_vector( y ) );
				cell_delta( y ) *= output_vector( y ) * learn_vector( y ) * cellActivationOutputDerivative( information_vector( y ) );
				output_delta( y ) *= m_cell_state( y ) * activationOutputDerivative( output_vector( y ) );
			}

			Vector new_delta;
			new_delta.setDimension( getInputCount() );

			for( unsigned int x = 0; x < getInputCount(); ++x ) {
				new_delta( x ) = 0.f;

				for( unsigned int y = 0; y < getOutputCount(); ++y ) {
					new_delta( x ) += forget_delta( y ) * m_forget_weights( y, x );
					new_delta( x ) += learn_delta( y ) * m_learn_weights( y, x );
					new_delta( x ) += cell_delta( y ) * m_cell_weights( y, x );
					new_delta( x ) += output_delta( y ) * m_output_weights( y, x );
				}
			}

			for( unsigned int y = 0; y < getOutputCount(); ++y ) {
				m_forget_bias( y ) -= mutability * forget_delta( y );
				m_learn_bias( y ) -= mutability * learn_delta( y );
				m_cell_bias( y ) -= mutability * cell_delta( y );
				m_output_bias( y ) -= mutability * output_delta( y );

				for( unsigned int x = 0; x < getInputCount(); ++x ) {
					m_forget_weights( y, x ) -= mutability * forget_delta( y ) * input( x );
					m_learn_weights( y, x ) -= mutability * learn_delta( y ) * input( x );
					m_cell_weights( y, x ) -= mutability * cell_delta( y ) * input( x );
					m_output_weights( y, x ) -= mutability * output_delta( y ) * input( x );
				}

				for( unsigned int x = 0; x < getOutputCount(); ++x ) {
					m_forget_state_weights( y, x ) -= mutability * forget_delta( y ) * output( x );
					m_learn_state_weights( y, x ) -= mutability * learn_delta( y ) * output( x );
					m_cell_state_weights( y, x ) -= mutability * cell_delta( y ) * output( x );
					m_output_state_weights( y, x ) -= mutability * output_delta( y ) * output( x );
				}
			}

			return new_delta;
		}

		virtual void resetState() {
			for( unsigned int i = 0; i < getOutputCount(); ++i ) {
				m_previous_output( i ) = 0.f;
				m_train_output( i ) = 0.f;
				m_cell_state( i ) = 0.f;
				m_train_state( i ) = 0.f;
			}
		}
};

#endif // LSTMLAYER_HPP
