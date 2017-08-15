#ifndef NETWORKLAYER_HPP
#define NETWORKLAYER_HPP

#include <string>
#include "json/json.h"
#include "Vector.hpp"

class NetworkLayer {
	private:
		unsigned int m_inputs;
		unsigned int m_outputs;

	protected:
		virtual void setSizeInternal( const unsigned int inputs, const unsigned int outputs ) = 0;

		virtual void loadFromJSONInternal( Json::Value& data_value ) = 0;
		virtual Json::Value saveToJSONInternal() = 0;
		virtual std::string getJSONTypeName() const = 0;

	public:
		void loadFromJSON( Json::Value& layer_value ) {
			setInputCount( layer_value[ "inputs" ].asUInt() );
			setOutputCount( layer_value[ "outputs" ].asUInt() );
			loadFromJSONInternal( layer_value[ "data" ] );
		}

		Json::Value saveToJSON() {
			Json::Value layer_object( Json::objectValue );
			layer_object[ "inputs" ] = Json::Value( getInputCount() );
			layer_object[ "outputs" ] = Json::Value( getOutputCount() );
			layer_object[ "type" ] = Json::Value( getJSONTypeName() );
			layer_object[ "data" ] = saveToJSONInternal();
			return layer_object;
		}

		unsigned int getInputCount() const {
			return m_inputs;
		}

		unsigned int getOutputCount() const {
			return m_outputs;
		}

		void setInputCount( const unsigned int inputs ) {
			m_inputs = inputs;
			setSizeInternal( inputs, getOutputCount() );
		}

		void setOutputCount( const unsigned int outputs ) {
			m_outputs = outputs;
			setSizeInternal( getInputCount(), outputs );
		}

		/**
		 * Propagate data through the network layer.
		 * @param input The input data to propagate.
		 * @return The output of the network layer.
		 */
		virtual Vector propagate( Vector input ) = 0;

		/**
		 * Train the network layer.
		 * @param input The input to the layer for training on.
		 * @param output The output of the layer being trained.
		 * @param delta The error from the next layer for training on.
		 * @param mutability The rate at which the layer is allowed to change.
		 * @return The error for passing into the next layer.
		 */
		virtual Vector train( Vector input, Vector output, Vector delta, float mutability = 0.05f ) = 0;

		/**
		 * Reset the state of the layer.
		 */
		virtual void resetState() {
		}
};

#endif // NETWORKLAYER_HPP
