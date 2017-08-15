#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

class Matrix {
	private:
		unsigned int m_width;
		unsigned int m_height;
		std::vector< float > m_values;

	public:
		/**
		 * Get the width of the matrix.
		 * @return The width of the matrix.
		 */
		unsigned int getWidth() const {
			return m_width;
		}

		/**
		 * Get the height of the matrix.
		 * @return The height of the matrix.
		 */
		unsigned int getHeight() const {
			return m_height;
		}

		/**
		 * Set the size of the matrix. Matrix contents are undefined afterwards.
		 * @param height The height of the matrix.
		 * @param width The width of the matrix.
		 */
		void setSize( unsigned int height, unsigned int width ) {
			m_width = width;
			m_height = height;
			m_values.resize( m_width * m_height );
		}

		/**
		 * Set the width of the matrix. Matrix contents are undefined afterwards.
		 * @param width The width of the matrix.
		 */
		void setWidth( unsigned int width ) {
			setSize( getHeight(), width );
		}

		/**
		 * Set the height of the matrix. Matrix contents are undefined afterwards.
		 * @param height The height of the matrix.
		 */
		void setHeight( unsigned int height ) {
			setSize( height, getWidth() );
		}

		/**
		 * Access the component of the matrix at the specified location.
		 * @param y The y index of the desired component.
		 * @param x The x index of the desired component.
		 * @return The component of the matrix being accessed.
		 */
		float& operator()( unsigned int y, unsigned int x ) {
			return m_values[ ( y % m_height ) * m_width + ( x % m_width ) ];
		}

		float* data() {
			return m_values.data();
		}
};

#endif // MATRIX_HPP
