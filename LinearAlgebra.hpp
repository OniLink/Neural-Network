#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <vector>

template< class T > class Vector {
	private:
		std::vector< T > values;

	public:
		/**
		 * Construct a vector.
		 * @param n The number of elements in the vector.
		 */
		Vector( unsigned int n ) :
			values( n, 0.f ) {
		}

		/**
		 * Access an element of the vector.
		 * @param i The index of the element to access.
		 * @return The element at the specified index.
		 */
		T& at( unsigned int i ) {
			return values[ i % values.size() ];
		}

		Vector< T > hadamard( Vector< T > rhs ) {
			Vector< T > output( getLength() );
			for( unsigned int i = 0; i < getLength(); ++i ) {
				output.at( i ) = at( i ) * rhs.at( i );
			}
			return output;
		}

		/**
		 * Get the number of elements in the vector.
		 * @return The number of elements in the vector.
		 */
		unsigned int getLength() const {
			return values.size();
		}

		std::vector< T >& getInternalData() {
			return values;
		}
};

template< class T > Vector< T > operator+( Vector< T > lhs, Vector< T > rhs ) {
	Vector< T > output( lhs.getLength() );

	if( lhs.getLength() != rhs.getLength() ) {
		return output;
	}

	for( unsigned int i = 0; i < lhs.getLength(); ++i ) {
		output.at( i ) = lhs.at( i ) + rhs.at( i );
	}

	return output;
}

template< class T > Vector< T > operator-( Vector< T > lhs, Vector< T > rhs ) {
	Vector< T > output( lhs.getLength() );

	if( lhs.getLength() != rhs.getLength() ) {
		return output;
	}

	for( unsigned int i = 0; i < lhs.getLength(); ++i ) {
		output.at( i ) = lhs.at( i ) - rhs.at( i );
	}

	return output;
}

template< class T > Vector< T > operator*( float lhs, Vector< T > rhs ) {
	Vector< T > output = rhs;

	for( unsigned int i = 0; i < output.getLength(); ++i ) {
		output.at( i ) *= lhs;
	}

	return output;
}

template< class T > Vector< T > operator*( Vector< T > lhs, float rhs ) {
	return rhs * lhs;
}

template< class T > class Matrix {
	private:
		unsigned int width;
		unsigned int height;
		std::vector< T > values;

	public:
		/**
		 * Construct a matrix.
		 * @param m The height of the matrix.
		 * @param n The width of the matrix.
		 */
		Matrix( unsigned int m, unsigned int n ) :
			width( n ),
			height( m ) {
			values.resize( m * n, 0.f );
		}

		/**
		 * Access an element of the matrix.
		 * @param i The vertical index of the matrix to access.
		 * @param j The horizontal index of the matrix to access.
		 * @return The element at the specified location.
		 */
		T& at( unsigned int i, unsigned int j ) {
			return values[ ( i % height ) * width + ( j % width ) ];
		}

		/**
		 * Get the width of the matrix (the number of inputs it takes).
		 * @return The width of the matrix.
		 */
		unsigned int getWidth() const {
			return width;
		}

		/**
		 * Get the height of the matrix (the number of outputs it gives).
		 * @return The height of the matrix.
		 */
		unsigned int getHeight() const {
			return height;
		}
};

template< class T > Matrix< T > operator+( Matrix< T > lhs, Matrix< T > rhs ) {
	Matrix< T > output( lhs.getHeight(), lhs.getWidth() );

	if( lhs.getHeight() != rhs.getHeight() || lhs.getWidth() != rhs.getWidth() ) {
		return output;
	}

	for( unsigned int y = 0; y < lhs.getHeight(); ++y ) {
		for( unsigned int x = 0; x < lhs.getWidth(); ++x ) {
			output.at( y, x ) = lhs.at( y, x ) + rhs.at( y, x );
		}
	}

	return output;
}

template< class T > Matrix< T > operator-( Matrix< T > lhs, Matrix< T > rhs ) {
	Matrix< T > output( lhs.getHeight(), lhs.getWidth() );

	if( lhs.getHeight() != rhs.getHeight() || lhs.getWidth() != rhs.getWidth() ) {
		return output;
	}

	for( unsigned int y = 0; y < lhs.getHeight(); ++y ) {
		for( unsigned int x = 0; x < lhs.getWidth(); ++x ) {
			output.at( y, x ) = lhs.at( y, x ) - rhs.at( y, x );
		}
	}

	return output;
}

template< class T > Matrix< T > operator*( Matrix< T > lhs, Matrix< T > rhs ) {
	Matrix< T > output( lhs.getHeight(), rhs.getWidth() );

	if( lhs.getWidth() != rhs.getHeight() ) {
		return output;
	}

	for( unsigned int y = 0; y < lhs.getHeight(); ++y ) {
		for( unsigned int x = 0; x < rhs.getWidth(); ++x ) {
			for( unsigned int i = 0; i < lhs.getWidth(); ++i ) {
				output.at( y, x ) += lhs.at( y, i ) * rhs.at( i, x );
			}
		}
	}

	return output;
}

template< class T > Vector< T > operator*( Matrix< T > lhs, Vector< T > rhs ) {
	Vector< T > output( lhs.getHeight() );

	if( lhs.getWidth() != rhs.getLength() ) {
		return output;
	}

	for( unsigned int y = 0; y < lhs.getHeight(); ++y ) {
		for( unsigned int x = 0; x < lhs.getWidth(); ++x ) {
			output.at( y ) += lhs.at( y, x ) * rhs.at( x );
		}
	}

	return output;
}

template< class T > Vector< T > operator*( Vector< T > lhs, Matrix< T > rhs ) {
	Vector< T > output( rhs.getHeight() );

	if( lhs.getLength() != rhs.getWidth() ) {
		return output;
	}

	for( unsigned int x = 0; x < rhs.getWidth(); ++x ) {
		for( unsigned int y = 0; y < rhs.getHeight(); ++y ) {
			output.at( x ) += lhs.at( y ) * rhs.at( y, x );
		}
	}

	return output;
}

template< class T > Matrix< T > operator*( Vector< T > lhs, Vector< T > rhs ) {
	Matrix< T > output( lhs.getLength(), rhs.getLength() );

	for( unsigned int y = 0; y < lhs.getLength(); ++y ) {
		for( unsigned int x = 0; x < rhs.getLength(); ++x ) {
			output.at( y, x ) = lhs.at( y ) * rhs.at( x );
		}
	}

	return output;
}

#endif
