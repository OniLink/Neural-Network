#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <vector>

class Vector {
	private:
		std::vector< float > m_values;

	public:
		/**
		 * Get the dimension of the vector.
		 * @return The vector dimension.
		 */
		unsigned int getDimension() const {
			return m_values.size();
		}

		/**
		 * Set the dimension of the vector. Its contents are undefined afterwards.
		 * @param size The new dimension of the vector.
		 */
		void setDimension( unsigned int size ) {
			m_values.resize( size );
		}

		/**
		 * Access the component of the vector at the specified index.
		 * @param index The index of the vector component to retrieve.
		 * @return The vector component being accessed.
		 */
		float& operator()( unsigned int index ) {
			return m_values[ index % getDimension() ];
		}

		float* data() {
			return m_values.data();
		}
};

#endif // VECTOR_HPP
