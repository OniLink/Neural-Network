#ifndef FFT_HPP
#define FFT_HPP

#include <complex>
#include <vector>

const float pi = std::acos( -1.f );

/**
 * Calculate the Fast Fourier Transform. Works fastest on inputs of length 2^N.
 * @param input The time-series input to the FFT.
 * @return The frequency-series output from the FFT.
 */
std::vector< std::complex< float > > fft( std::vector< std::complex< float > > input ) {
	std::vector< std::complex< float > > output( input.size(), std::complex< float >( 0.f, 0.f ) );

	if( input.size() <= 1 ) {
		output = input;
	} else if( input.size() % 2 == 0 ) {
		unsigned int partial_size = input.size() / 2;
		std::vector< std::complex< float > > left( partial_size );
		std::vector< std::complex< float > > right( partial_size );

		for( unsigned int i = 0; i < partial_size; ++i ) {
			left[ i ] = input[ 2 * i ];
			right[ i ] = input[ 2 * i + 1 ];
		}

		left = fft( left );
		right = fft( right );

		float partial_frequency = -2.f * pi / static_cast< float >( input.size() );

		for( unsigned int i = 0; i < partial_size; ++i ) {
			std::complex< float > twiddle = std::exp( std::complex< float >( 0.f, partial_frequency * i ) );
			output[ i ] = left[ i ] + twiddle * right[ i ];
			output[ i + partial_size ] = left[ i ] - twiddle * right[ i ];
		}
	} else {
		float partial_frequency = -2.f * pi / static_cast< float >( input.size() );

		for( unsigned int i = 0; i < input.size(); ++i ) {
			for( unsigned int j = 0; j < input.size(); ++j ) {
				std::complex< float > twiddle = std::exp( std::complex< float >( 0.f, partial_frequency * i * j ) );
				output[ i ] += twiddle * input[ j ];
			}
		}
	}

	return output;
}

/**
 * Calculate the inverse Fast Fourier Transform. Works fastest on inputs of length 2^N.
 * @param input The frequency-series input to the IFFT.
 * @return The time-series output from the IFFT.
 */
std::vector< std::complex< float > > ifft( std::vector< std::complex< float > > input ) {
	for( unsigned int i = 0; i < input.size(); ++i ) {
		input[ i ] = std::conj( input[ i ] );
	}

	input = fft( input );

	for( unsigned int i = 0; i < input.size(); ++i ) {
		input[ i ] = std::conj( input[ i ] ) / std::complex< float >( input.size(), 0.f );
	}

	return input;
}

/**
 * Calculate the short-time Fourier Transform of a time sequence.
 * @param data The time sequence to be transformed. Algorithm works best if its length is a multiple of the step size.
 * @param step_size The number of samples to use in each slice. Recommended to be a power of two greater than (sample rate / 40).
 * @return The set of frequency space time sliced data. Each covers a time step equal to the step size.
 */
std::vector< std::vector< std::complex< float > > > stft( std::vector< std::complex< float > > data, unsigned int step_size = 2048 ) {
	std::vector< std::vector< std::complex< float > > > output_chunks;

	unsigned int data_size = data.size();
	if( data_size % step_size != 0 ) {
		data_size += step_size - ( data_size % step_size );
	}
	data.resize( data_size );

	for( unsigned int read_pos = 0; read_pos + step_size <= data.size(); read_pos += step_size ) {
		std::vector< std::complex< float > > data_input( 2 * step_size, std::complex< float >( 0.f, 0.f ) );
		for( unsigned int i = 0; i < step_size; ++i ) {
			data_input[ i ] = data[ read_pos + i ];
		}

		std::vector< std::complex< float > > data_output = fft( data_input );
		output_chunks.push_back( data_output );
	}

	return output_chunks;
}

/**
 * Inverts the short-time Fourier Transform to turn a frequency sequence into time sequence.
 * @param chunks The chunks of frequency data to be turned into time sequence data. Individual chunks must have a length multiple of two.
 * @return The time sequence corresponding to the frequency data.
 */
std::vector< std::complex< float > > istft( std::vector< std::vector< std::complex< float > > > chunks ) {
	unsigned int length = 0;
	for( unsigned int i = 0; i < chunks.size(); ++i ) {
		length += chunks[ i ].size();
	}
	length /= 2;

	std::vector< std::complex< float > > output_series( length, std::complex< float >( 0.f, 0.f ) );

	unsigned int write_pos = 0;

	for( unsigned int chunk_num = 0; chunk_num < chunks.size(); ++chunk_num ) {
		std::vector< std::complex< float > > chunk_time = ifft( chunks[ chunk_num ] );

		for( unsigned int i = 0; i < chunk_time.size() && write_pos + i < length; ++i ) {
			output_series[ write_pos + i ] = chunk_time[ i ];
		}

		write_pos += chunks[ chunk_num ].size() / 2;
	}

	return output_series;
}

#endif // FFT_HPP
