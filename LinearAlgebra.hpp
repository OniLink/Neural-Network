#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <iostream>
#include <vector>
#include "cl.hpp"



namespace lasettings {
bool use_opencl = false;
cl::Context opencl_context;
cl::CommandQueue opencl_queue;

const std::string code_vector_hadamard =
	"__kernel void vector_hadamard( __global const float* left, __global const float* right, __global float* out ) {\n"
	"	const int index = get_global_id( 0 );\n"
	"	out[ index ] = left[ index ] * right[ index ];\n"
	"}\n";

const std::string code_vector_add =
	"__kernel void vector_add( __global const float* left, __global const float* right, __global float* out ) {\n"
	"	const int index = get_global_id( 0 );\n"
	"	out[ index ] = left[ index ] + right[ index ];\n"
	"}\n";

const std::string code_vector_sub =
	"__kernel void vector_sub( __global const float* left, __global const float* right, __global float* out ) {\n"
	"	const int index = get_global_id( 0 );\n"
	"	out[ index ] = left[ index ] - right[ index ];\n"
	"}\n";

const std::string code_vector_scalar_mul =
	"__kernel void vector_scalar_mul( const float left, __global const float* right, __global float* out ) {\n"
	"	const int index = get_global_id( 0 );\n"
	"	out[ index ] = left * right[ index ];\n"
	"}\n";

const std::string code_matrix_add =
	"__kernel void matrix_add( const unsigned int W, __global const float* left, __global const float* right, __global float* out ) {\n"
	"	const int y = get_global_id( 0 );\n"
	"	const int x = get_global_id( 1 );\n"
	"	out[ y * W + x ] = left[ y * W + x ] + right[ y * W + x ];\n"
	"}\n";

const std::string code_matrix_sub =
	"__kernel void matrix_sub( const unsigned int W, __global const float* left, __global const float* right, __global float* out ) {\n"
	"	const int y = get_global_id( 0 );\n"
	"	const int x = get_global_id( 1 );\n"
	"	out[ y * W + x ] = left[ y * W + x ] - right[ y * W + x ];\n"
	"}\n";

const std::string code_matrix_matrix_mul =
	"__kernel void matrix_matrix_mul( const unsigned int W, const unsigned int K,\n"
	"								  __global const float* left, __global const float* right, __global float* out ) {\n"
	"	const int y = get_global_id( 0 );\n"
	"	const int x = get_global_id( 1 );\n"
	"	float accum = 0.f;\n"
	"	for( unsigned int k = 0; k < K; ++k ) {\n"
	"		accum += left[ y * W + k ] * right[ k * W + x ];\n"
	"	}\n"
	"	out[ y * W + x ] = accum;\n"
	"}\n";

const std::string code_matrix_vector_mul =
	"__kernel void matrix_vector_mul( const unsigned int W, __global const float* left, __global const float* right, __global float* out ) {\n"
	"	const int y = get_global_id( 0 );\n"
	"	float accum = 0.f;\n"
	"	for( unsigned int k = 0; k < W; ++k ) {\n"
	"		accum += left[ y * W + k ] * right[ k ];\n"
	"	}\n"
	"	out[ y ] = accum;\n"
	"}\n";

const std::string code_vector_matrix_mul =
	"__kernel void vector_matrix_mul( const unsigned int H, const unsigned int W,\n"
	"								  __global const float* left, __global const float* right, __global float* out ) {\n"
	"	const int x = get_global_id( 0 );\n"
	"	float accum = 0.f;\n"
	"	for( unsigned int k = 0; k < H; ++k ) {\n"
	"		accum += left[ k ] * right[ k * W + x ];\n"
	"	}\n"
	"	out[ x ] = accum;\n"
	"}\n";

const std::string code_vector_vector_mul =
	"__kernel void vector_vector_mul( const unsigned int W, __global const float* left, __global const float* right, __global float* out ) {\n"
	"	const int y = get_global_id( 0 );\n"
	"	const int x = get_global_id( 1 );\n"
	"	out[ y * W + x ] = left[ y ] * right[ x ];\n"
	"}\n";

cl::Kernel kernel_vector_hadamard;
cl::Kernel kernel_vector_add;
cl::Kernel kernel_vector_sub;
cl::Kernel kernel_vector_scalar_mul;

cl::Kernel kernel_matrix_add;
cl::Kernel kernel_matrix_sub;
cl::Kernel kernel_matrix_matrix_mul;
cl::Kernel kernel_matrix_vector_mul;
cl::Kernel kernel_vector_matrix_mul;
cl::Kernel kernel_vector_vector_mul;

bool setupOpenCL() {
	// Set up the OpenCL Context
	std::vector< cl::Platform > platforms;
	cl::Platform::get( &platforms );
	if( platforms.size() == 0 ) {
		use_opencl = false;
		std::cout << "OpenCL Initialization Error: No platforms found\n";
		return false;
	}
	cl::Platform platform = platforms[ 0 ];

	std::vector< cl::Device > devices;
	platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );
	if( devices.size() == 0 ) {
		use_opencl = false;
		std::cout << "OpenCL Initialization Error: No devices found\n";
		return false;
	}
	cl::Device device = devices[ 0 ];

	opencl_context = cl::Context( { device } );

	// Set up the linear algebra program
	cl::Program::Sources program_sources;
	program_sources.push_back( { code_vector_hadamard.data(), code_vector_hadamard.length() } );
	program_sources.push_back( { code_vector_add.data(), code_vector_add.length() } );
	program_sources.push_back( { code_vector_sub.data(), code_vector_sub.length() } );
	program_sources.push_back( { code_vector_scalar_mul.data(), code_vector_scalar_mul.length() } );
	program_sources.push_back( { code_matrix_add.data(), code_matrix_add.length() } );
	program_sources.push_back( { code_matrix_sub.data(), code_matrix_sub.length() } );
	program_sources.push_back( { code_matrix_matrix_mul.data(), code_matrix_matrix_mul.length() } );
	program_sources.push_back( { code_matrix_vector_mul.data(), code_matrix_vector_mul.length() } );
	program_sources.push_back( { code_vector_matrix_mul.data(), code_vector_matrix_mul.length() } );
	program_sources.push_back( { code_vector_vector_mul.data(), code_vector_vector_mul.length() } );

	cl::Program program( opencl_context, program_sources );
	if( program.build( { device } ) != CL_SUCCESS ) {
		use_opencl = false;
		std::cout << "OpenCL Initialization Error: Failed to build OpenCL program\n";
		std::cout << "Build Log: " << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device ) << std::endl;
		return false;
	}

	kernel_vector_hadamard = cl::Kernel( program, "vector_hadamard" );
	kernel_vector_add = cl::Kernel( program, "vector_add" );
	kernel_vector_sub = cl::Kernel( program, "vector_sub" );
	kernel_vector_scalar_mul = cl::Kernel( program, "vector_scalar_mul" );
	kernel_matrix_add = cl::Kernel( program, "matrix_add" );
	kernel_matrix_sub = cl::Kernel( program, "matrix_sub" );
	kernel_matrix_matrix_mul = cl::Kernel( program, "matrix_matrix_mul" );
	kernel_matrix_vector_mul = cl::Kernel( program, "matrix_vector_mul" );
	kernel_vector_matrix_mul = cl::Kernel( program, "vector_matrix_mul" );
	kernel_vector_vector_mul = cl::Kernel( program, "vector_vector_mul" );

	// Set up the command queue
	opencl_queue = cl::CommandQueue( opencl_context, device );

	// Done
	use_opencl = true;
	return true;
}

}

class Vector {
	private:
		std::vector< float > values;

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
		float& at( unsigned int i ) {
			return values[ i % values.size() ];
		}

		Vector hadamard( Vector rhs ) {
			Vector output( getLength() );

			if( getLength() != rhs.getLength() ) {
				return output;
			}

			if( !lasettings::use_opencl ) {
				for( unsigned int i = 0; i < getLength(); ++i ) {
					output.at( i ) = at( i ) * rhs.at( i );
				}
			} else {
				const unsigned int element_count = getLength();
				cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * element_count );
				lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ) * element_count, getInternalData().data() );
				cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * element_count );
				lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * element_count, rhs.getInternalData().data() );
				cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * element_count );

				lasettings::kernel_vector_hadamard.setArg( 0, left_buffer );
				lasettings::kernel_vector_hadamard.setArg( 1, right_buffer );
				lasettings::kernel_vector_hadamard.setArg( 2, output_buffer );

				lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_vector_hadamard, cl::NullRange, cl::NDRange( element_count ), cl::NullRange );
				lasettings::opencl_queue.finish();
				lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * element_count, output.getInternalData().data() );
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

		std::vector< float >& getInternalData() {
			return values;
		}
};

inline Vector operator+( Vector lhs, Vector rhs ) {
	Vector output( lhs.getLength() );

	if( lhs.getLength() != rhs.getLength() ) {
		return output;
	}

	if( !lasettings::use_opencl ) {
		for( unsigned int i = 0; i < lhs.getLength(); ++i ) {
			output.at( i ) = lhs.at( i ) + rhs.at( i );
		}
	} else {
		const unsigned int element_count = lhs.getLength();
		cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * element_count );
		lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ) * element_count, lhs.getInternalData().data() );
		cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * element_count );
		lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * element_count, rhs.getInternalData().data() );
		cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * element_count );

		lasettings::kernel_vector_add.setArg( 0, left_buffer );
		lasettings::kernel_vector_add.setArg( 1, right_buffer );
		lasettings::kernel_vector_add.setArg( 2, output_buffer );

		lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_vector_add, cl::NullRange, cl::NDRange( element_count ), cl::NullRange );
		lasettings::opencl_queue.finish();
		lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * element_count, output.getInternalData().data() );
	}

	return output;
}

inline Vector operator-( Vector lhs, Vector rhs ) {
	Vector output( lhs.getLength() );

	if( lhs.getLength() != rhs.getLength() ) {
		return output;
	}

	if( !lasettings::use_opencl ) {
		for( unsigned int i = 0; i < lhs.getLength(); ++i ) {
			output.at( i ) = lhs.at( i ) - rhs.at( i );
		}
	} else {
		const unsigned int element_count = lhs.getLength();
		cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * element_count );
		lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ) * element_count, lhs.getInternalData().data() );
		cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * element_count );
		lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * element_count, rhs.getInternalData().data() );
		cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * element_count );

		lasettings::kernel_vector_sub.setArg( 0, left_buffer );
		lasettings::kernel_vector_sub.setArg( 1, right_buffer );
		lasettings::kernel_vector_sub.setArg( 2, output_buffer );

		lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_vector_sub, cl::NullRange, cl::NDRange( element_count ), cl::NullRange );
		lasettings::opencl_queue.finish();
		lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * element_count, output.getInternalData().data() );
	}

	return output;
}

inline Vector operator*( float lhs, Vector rhs ) {
	Vector output = rhs;

	if( !lasettings::use_opencl ) {
		for( unsigned int i = 0; i < output.getLength(); ++i ) {
			output.at( i ) *= lhs;
		}
	} else {
		const unsigned int element_count = rhs.getLength();
		cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) );
		lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ), &lhs );
		cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * element_count );
		lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * element_count, rhs.getInternalData().data() );
		cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * element_count );

		lasettings::kernel_vector_scalar_mul.setArg( 0, left_buffer );
		lasettings::kernel_vector_scalar_mul.setArg( 1, right_buffer );
		lasettings::kernel_vector_scalar_mul.setArg( 2, output_buffer );

		lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_vector_scalar_mul, cl::NullRange, cl::NDRange( element_count ), cl::NullRange );
		lasettings::opencl_queue.finish();
		lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * element_count, output.getInternalData().data() );
	}

	return output;
}

inline Vector operator*( Vector lhs, float rhs ) {
	return rhs * lhs;
}

class Matrix {
	private:
		unsigned int width;
		unsigned int height;
		std::vector< float > values;

	public:
		/**
		 * Construct a matrix.
		 * @param m The height of the matrix.
		 * @param n The width of the matrix.
		 */
		Matrix( unsigned int m, unsigned int n ) :
			width( n ),
			height( m ),
			values( m * n, 0.f ) {
		}

		/**
		 * Access an element of the matrix.
		 * @param i The vertical index of the matrix to access.
		 * @param j The horizontal index of the matrix to access.
		 * @return The element at the specified location.
		 */
		float& at( unsigned int i, unsigned int j ) {
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

		std::vector< float >& getInternalData() {
			return values;
		}
};

inline Matrix operator+( Matrix lhs, Matrix rhs ) {
	Matrix output( lhs.getHeight(), lhs.getWidth() );

	if( lhs.getHeight() != rhs.getHeight() || lhs.getWidth() != rhs.getWidth() ) {
		return output;
	}

	if( !lasettings::use_opencl ) {
		for( unsigned int y = 0; y < lhs.getHeight(); ++y ) {
			for( unsigned int x = 0; x < lhs.getWidth(); ++x ) {
				output.at( y, x ) = lhs.at( y, x ) + rhs.at( y, x );
			}
		}
	} else {
		const unsigned int matrix_width = lhs.getWidth();
		const unsigned int matrix_height = lhs.getHeight();
		cl::Buffer width_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( unsigned int ) );
		lasettings::opencl_queue.enqueueWriteBuffer( width_buffer, CL_FALSE, 0, sizeof( unsigned int ), &matrix_width );

		cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_width * matrix_height );
		lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ) * matrix_width * matrix_height, lhs.getInternalData().data() );
		cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_width * matrix_height );
		lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * matrix_width * matrix_height, rhs.getInternalData().data() );
		cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * matrix_width * matrix_height );

		lasettings::kernel_matrix_add.setArg( 0, width_buffer );
		lasettings::kernel_matrix_add.setArg( 1, left_buffer );
		lasettings::kernel_matrix_add.setArg( 2, right_buffer );
		lasettings::kernel_matrix_add.setArg( 3, output_buffer );

		lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_matrix_add, cl::NullRange, cl::NDRange( matrix_height, matrix_width ), cl::NullRange );
		lasettings::opencl_queue.finish();
		lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * matrix_width * matrix_height, output.getInternalData().data() );
	}

	return output;
}

inline Matrix operator-( Matrix lhs, Matrix rhs ) {
	Matrix output( lhs.getHeight(), lhs.getWidth() );

	if( lhs.getHeight() != rhs.getHeight() || lhs.getWidth() != rhs.getWidth() ) {
		return output;
	}

	if( !lasettings::use_opencl ) {
		for( unsigned int y = 0; y < lhs.getHeight(); ++y ) {
			for( unsigned int x = 0; x < lhs.getWidth(); ++x ) {
				output.at( y, x ) = lhs.at( y, x ) - rhs.at( y, x );
			}
		}
	} else {
		const unsigned int matrix_width = lhs.getWidth();
		const unsigned int matrix_height = lhs.getHeight();
		cl::Buffer width_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( unsigned int ) );
		lasettings::opencl_queue.enqueueWriteBuffer( width_buffer, CL_FALSE, 0, sizeof( unsigned int ), &matrix_width );

		cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_width * matrix_height );
		lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ) * matrix_width * matrix_height, lhs.getInternalData().data() );
		cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_width * matrix_height );
		lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * matrix_width * matrix_height, rhs.getInternalData().data() );
		cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * matrix_width * matrix_height );

		lasettings::kernel_matrix_sub.setArg( 0, width_buffer );
		lasettings::kernel_matrix_sub.setArg( 1, left_buffer );
		lasettings::kernel_matrix_sub.setArg( 2, right_buffer );
		lasettings::kernel_matrix_sub.setArg( 3, output_buffer );

		lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_matrix_sub, cl::NullRange, cl::NDRange( matrix_height, matrix_width ), cl::NullRange );
		lasettings::opencl_queue.finish();
		lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * matrix_width * matrix_height, output.getInternalData().data() );
	}

	return output;
}

inline Matrix operator*( Matrix lhs, Matrix rhs ) {
	Matrix output( lhs.getHeight(), rhs.getWidth() );

	if( lhs.getWidth() != rhs.getHeight() ) {
		return output;
	}

	if( !lasettings::use_opencl ) {
		for( unsigned int y = 0; y < lhs.getHeight(); ++y ) {
			for( unsigned int x = 0; x < rhs.getWidth(); ++x ) {
				for( unsigned int i = 0; i < lhs.getWidth(); ++i ) {
					output.at( y, x ) += lhs.at( y, i ) * rhs.at( i, x );
				}
			}
		}
	} else {
		const unsigned int matrix_width = rhs.getWidth();
		const unsigned int matrix_height = lhs.getHeight();
		const unsigned int matrix_shared = lhs.getWidth();
		cl::Buffer width_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( unsigned int ) );
		lasettings::opencl_queue.enqueueWriteBuffer( width_buffer, CL_FALSE, 0, sizeof( unsigned int ), &matrix_width );
		cl::Buffer shared_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( unsigned int ) );
		lasettings::opencl_queue.enqueueWriteBuffer( shared_buffer, CL_FALSE, 0, sizeof( unsigned int ), &matrix_shared );

		cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_shared * matrix_height );
		lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ) * matrix_shared * matrix_height, lhs.getInternalData().data() );
		cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_width * matrix_shared );
		lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * matrix_width * matrix_shared, rhs.getInternalData().data() );
		cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * matrix_width * matrix_height );

		lasettings::kernel_matrix_matrix_mul.setArg( 0, width_buffer );
		lasettings::kernel_matrix_matrix_mul.setArg( 1, shared_buffer );
		lasettings::kernel_matrix_matrix_mul.setArg( 2, left_buffer );
		lasettings::kernel_matrix_matrix_mul.setArg( 3, right_buffer );
		lasettings::kernel_matrix_matrix_mul.setArg( 4, output_buffer );

		lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_matrix_matrix_mul, cl::NullRange, cl::NDRange( matrix_height, matrix_width ), cl::NullRange );
		lasettings::opencl_queue.finish();
		lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * matrix_width * matrix_height, output.getInternalData().data() );
	}

	return output;
}

inline Vector operator*( Matrix lhs, Vector rhs ) {
	Vector output( lhs.getHeight() );

	if( lhs.getWidth() != rhs.getLength() ) {
		return output;
	}

	if( !lasettings::use_opencl ) {
		for( unsigned int y = 0; y < lhs.getHeight(); ++y ) {
			for( unsigned int x = 0; x < lhs.getWidth(); ++x ) {
				output.at( y ) += lhs.at( y, x ) * rhs.at( x );
			}
		}
	} else {
		const unsigned int matrix_width = lhs.getWidth();
		const unsigned int matrix_height = lhs.getHeight();
		cl::Buffer width_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( unsigned int ) );
		lasettings::opencl_queue.enqueueWriteBuffer( width_buffer, CL_FALSE, 0, sizeof( unsigned int ), &matrix_width );

		cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_width * matrix_height );
		lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ) * matrix_width * matrix_height, lhs.getInternalData().data() );
		cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_width );
		lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * matrix_width, rhs.getInternalData().data() );
		cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * matrix_height );

		lasettings::kernel_matrix_vector_mul.setArg( 0, width_buffer );
		lasettings::kernel_matrix_vector_mul.setArg( 1, left_buffer );
		lasettings::kernel_matrix_vector_mul.setArg( 2, right_buffer );
		lasettings::kernel_matrix_vector_mul.setArg( 3, output_buffer );

		lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_matrix_vector_mul, cl::NullRange, cl::NDRange( matrix_height ), cl::NullRange );
		lasettings::opencl_queue.finish();
		lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * matrix_height, output.getInternalData().data() );
	}

	return output;
}

inline Vector operator*( Vector lhs, Matrix rhs ) {
	Vector output( rhs.getWidth() );

	if( lhs.getLength() != rhs.getHeight() ) {
		return output;
	}

	if( !lasettings::use_opencl ) {
		for( unsigned int x = 0; x < rhs.getWidth(); ++x ) {
			for( unsigned int y = 0; y < rhs.getHeight(); ++y ) {
				output.at( x ) += lhs.at( y ) * rhs.at( y, x );
			}
		}
	} else {
		const unsigned int matrix_width = rhs.getWidth();
		const unsigned int matrix_height = rhs.getHeight();
		cl::Buffer height_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( unsigned int ) );
		lasettings::opencl_queue.enqueueWriteBuffer( height_buffer, CL_FALSE, 0, sizeof( unsigned int ), &matrix_height );
		cl::Buffer width_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( unsigned int ) );
		lasettings::opencl_queue.enqueueWriteBuffer( width_buffer, CL_FALSE, 0, sizeof( unsigned int ), &matrix_width );

		cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_height );
		lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ) * matrix_height, lhs.getInternalData().data() );
		cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_width * matrix_height );
		lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * matrix_width * matrix_height, rhs.getInternalData().data() );
		cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * matrix_width );

		lasettings::kernel_vector_matrix_mul.setArg( 0, height_buffer );
		lasettings::kernel_vector_matrix_mul.setArg( 1, width_buffer );
		lasettings::kernel_vector_matrix_mul.setArg( 2, left_buffer );
		lasettings::kernel_vector_matrix_mul.setArg( 3, right_buffer );
		lasettings::kernel_vector_matrix_mul.setArg( 4, output_buffer );

		lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_vector_matrix_mul, cl::NullRange, cl::NDRange( matrix_width ), cl::NullRange );
		lasettings::opencl_queue.finish();
		lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * matrix_width, output.getInternalData().data() );
	}

	return output;
}

inline Matrix operator*( Vector lhs, Vector rhs ) {
	Matrix output( lhs.getLength(), rhs.getLength() );

	if( !lasettings::use_opencl ) {
		for( unsigned int y = 0; y < lhs.getLength(); ++y ) {
			for( unsigned int x = 0; x < rhs.getLength(); ++x ) {
				output.at( y, x ) = lhs.at( y ) * rhs.at( x );
			}
		}
	} else {
		unsigned int matrix_width = rhs.getLength();
		unsigned int matrix_height = lhs.getLength();

		cl::Buffer width_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( unsigned int ) );
		lasettings::opencl_queue.enqueueWriteBuffer( width_buffer, CL_FALSE, 0, sizeof( unsigned int ), &matrix_width );

		cl::Buffer left_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_height );
		lasettings::opencl_queue.enqueueWriteBuffer( left_buffer, CL_FALSE, 0, sizeof( float ) * matrix_height, lhs.getInternalData().data() );
		cl::Buffer right_buffer( lasettings::opencl_context, CL_MEM_READ_ONLY, sizeof( float ) * matrix_width );
		lasettings::opencl_queue.enqueueWriteBuffer( right_buffer, CL_FALSE, 0, sizeof( float ) * matrix_width, rhs.getInternalData().data() );
		cl::Buffer output_buffer( lasettings::opencl_context, CL_MEM_WRITE_ONLY, sizeof( float ) * matrix_width * matrix_height );

		lasettings::kernel_vector_vector_mul.setArg( 0, width_buffer );
		lasettings::kernel_vector_vector_mul.setArg( 1, left_buffer );
		lasettings::kernel_vector_vector_mul.setArg( 2, right_buffer );
		lasettings::kernel_vector_vector_mul.setArg( 3, output_buffer );

		lasettings::opencl_queue.enqueueNDRangeKernel( lasettings::kernel_vector_vector_mul, cl::NullRange, cl::NDRange( matrix_height, matrix_width ), cl::NullRange );
		lasettings::opencl_queue.finish();
		lasettings::opencl_queue.enqueueReadBuffer( output_buffer, CL_TRUE, 0, sizeof( float ) * matrix_width * matrix_height, output.getInternalData().data() );
	}

	return output;
}

#endif
