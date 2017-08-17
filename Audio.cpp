#include <fstream>
#include <iostream>
#include <memory>
#include <SFML/Audio/InputSoundFile.hpp>
#include <SFML/Audio/OutputSoundFile.hpp>

#include "json/json.h"
#include "FFT.hpp"
#include "FeedForwardLayer.hpp"
#include "LSTMLayer.hpp"
#include "NeuralNetwork.hpp"

NeuralNetwork network;
unsigned int channel_count = 2;
unsigned int sample_rate = 48000;
unsigned int step_size = 2048;
unsigned int chunk_size = 4096;

void setupNetwork() {
	unsigned int layer_count = 0;
	std::cout << "Enter number of network layers: ";
	std::cin >> layer_count;

	std::cout << "Available Network Layer Types:\n";
	std::cout << "01 - Feed Forward\n";
	std::cout << "02 - Long Short Term Memory\n";

	for( unsigned int i = 0; i < layer_count; ++i ) {
		unsigned int type = 0;
		bool valid = true;
		NetworkLayer* layer = nullptr;
		std::cout << "Enter type of layer " << i+1 << '/' << layer_count << ": ";
		do {
			std::cin >> type;
			valid = true;

			switch( type ) {
				case 1: {
						layer = new FeedForwardLayer;
					}
					break;

				case 2: {
						layer = new LSTMLayer;
					}
					break;

				default:
					valid = false;
					std::cout << "Invalid type. Try again: ";
					break;
			}
		} while( !valid );

		if( i == 0 ) {
			layer->setInputCount( 0 );
		}

		unsigned int outputs = channel_count * step_size * 2;

		if( i != layer_count - 1 ) {
			std::cout << "Enter the number of outputs to the layer: ";
			std::cin >> outputs;
		}

		layer->setOutputCount( outputs );

		network.addLayer( layer );
	}

	std::cout << "Network built.\n";
}

void instructGenerate() {
	network.resetState();

	std::string output_filename;
	std::cout << "Give an output filename: ";
	std::cin >> output_filename;

	unsigned int length_chunks;
	std::cout << "Give song length in seconds: ";
	std::cin >> length_chunks;
	length_chunks *= sample_rate / step_size;

	sf::OutputSoundFile output_file;
	if( !output_file.openFromFile( output_filename, sample_rate, channel_count ) ) {
		std::cout << "Failed to open output file\n";
		return;
	}

	Vector input;
	input.setDimension( 0 );

	// Multiple Layers
	// Channel Count< Chunk Count< Frequencies< Magnitude > > >
	std::vector< std::vector< std::vector< std::complex< float > > > > output_chunks( channel_count );

	for( unsigned int i = 0; i < length_chunks; ++i ) {
		std::cout << i << '/' << length_chunks << " chunks rendered\n";

		Vector sample = network.propagate( input );

		// Channel Count< Frequencies< Magnitude > >
		std::vector< std::vector< std::complex< float > > > chunk_data( channel_count, std::vector< std::complex< float > >( chunk_size ) );

		for( unsigned int j = 0; j < step_size; ++j ) {
			for( unsigned int c = 0; c < channel_count; ++c ) {
				unsigned int sample_pos = 2 * ( channel_count * j + c );
				chunk_data[ c ][ j ] = std::complex< float >( sample( sample_pos ), sample( sample_pos + 1 ) );
			}
		}

		for( unsigned int j = 1; j < step_size; ++j ) {
			for( unsigned int c = 0; c < channel_count; ++c ) {
				chunk_data[ c ][ chunk_size - j ] = std::conj( chunk_data[ c ][ j ] );
			}
		}

		for( unsigned int c = 0; c < channel_count; ++c ) {
			output_chunks[ c ].push_back( chunk_data[ c ] );
		}
	}

	std::cout << "Converting from frequency to time\n";

	std::vector< std::vector< std::complex< float > > > output_time_series( channel_count );

	for( unsigned int c = 0; c < channel_count; ++c ) {
		output_time_series[ c ] = istft( output_chunks[ c ] );
	}

	std::cout << "Renormalizing\n";

	float max_amp = 0.f;
	for( unsigned int c = 0; c < channel_count; ++c ) {
		for( unsigned int i = 0; i < output_time_series[ c ].size(); ++i ) {
			if( std::abs( output_time_series[ c ][ i ].real() ) > max_amp ) {
				max_amp = std::abs( output_time_series[ c ][ i ].real() );
			}
		}
	}

	std::cout << "Converting to 16-bit PCM format\n";

	std::vector< sf::Int16 > output_samples;

	for( unsigned int i = 0; i < length_chunks * step_size; ++i ) {
		for( unsigned int c = 0; c < channel_count; ++c ) {
			output_samples.push_back( ( output_time_series[ c ][ i ].real() / max_amp ) * 32768 );
		}
	}

	std::cout << "Writing to file\n";

	output_file.write( output_samples.data(), output_samples.size() );
}

void instructHelp() {
	std::cout << "List of commands:\n";
	std::cout << "g - Generate an output file\n";
	std::cout << "h - Print this help menu\n";
	std::cout << "l - Load the neural network from a file\n";
	std::cout << "q - Quit the application\n";
	std::cout << "s - Save the neural network to a file\n";
	std::cout << "t - Train on an audio file\n";
}

void instructLoad() {
	std::string filename;
	std::cout << "Enter network filename: ";
	std::cin >> filename;

	std::ifstream input_file( filename );
	if( !input_file.is_open() || !input_file.good() ) {
		std::cout << "Failed to open file \"" << filename << "\" for loading\n";
		return;
	}

	std::string input_data;

	do {
		char in = input_file.get();

		if( input_file.good() ) {
			input_data += in;
		}
	} while( input_file.good() );

	if( !input_file.eof() ) {
		std::cout << "Failed to read file\n";
		return;
	}

	Json::CharReaderBuilder reader_builder;
	reader_builder[ "collectComments" ] = false;

	std::unique_ptr< Json::CharReader > reader( reader_builder.newCharReader() );

	Json::Value root;

	if( !reader->parse( input_data.data(), input_data.data() + input_data.size(), &root, nullptr ) ) {
		std::cout << "Unable to parse JSON file\n";
		return;
	}

	sample_rate = root[ "sample-rate" ].asUInt();
	channel_count = root[ "channels" ].asUInt();
	step_size = root[ "stft-size" ].asUInt();

	network.loadFromJSON( root[ "layers" ] );
}

void instructSave() {
	std::cout << "Creating JSON data\n";
	Json::Value root( Json::objectValue );
	root[ "sample-rate" ] = Json::Value( sample_rate );
	root[ "channels" ] = Json::Value( channel_count );
	root[ "stft-size" ] = Json::Value( step_size );
	root[ "layers" ] = network.saveToJSON();

	Json::StreamWriterBuilder writer_builder;
	writer_builder[ "commentStyle" ] = "None";
	writer_builder[ "indentation" ] = "";
	writer_builder[ "enableYAMLCompatibility" ] = true;

	std::unique_ptr< Json::StreamWriter > writer( writer_builder.newStreamWriter() );

	std::string filename;
	std::cout << "Enter a filename for the network: ";
	std::cin >> filename;

	std::ofstream output_file( filename );
	if( !output_file.is_open() || !output_file.good() ) {
		std::cout << "Failed to open file \"" << filename << "\" for saving\n";
		return;
	}

	writer->write( root, &output_file );

	output_file.flush();

	if( !output_file.good() ) {
		std::cout << "Failed to write to disk\n";
	}

	output_file.close();
}

std::vector< float > readSamples( sf::InputSoundFile& input_file ) {
	std::vector< float > training_samples( 0 );
	sf::Int16 samples_in[ 1024 ];
	sf::Uint64 read_count = 0;

	do {
		read_count = input_file.read( samples_in, 1024 );
		for( unsigned int i = 0; i < read_count; ++i ) {
			training_samples.push_back( static_cast< float >( samples_in[ i ] ) / 32768.f );
		}
	} while( read_count > 0 );

	return training_samples;
}

std::vector< float > readTrainingFile( std::string filename ) {
	sf::InputSoundFile training_file;
	if( !training_file.openFromFile( filename ) ) {
		std::cout << "Invalid file\n";
		return std::vector< float >( 0 );
	}

	if( training_file.getChannelCount() != channel_count ) {
		std::cout << "Channel count in file does not match channel count in network\n";
		return std::vector< float >( 0 );
	}

	return readSamples( training_file );
}

void instructTrain() {
	std::string training_filename;
	std::cout << "Enter filename of training file: ";
	std::cin >> training_filename;

	std::cout << "Reading file\n";

	std::vector< float > training_samples = readTrainingFile( training_filename );

	if( training_samples.size() == 0 ) {
		return;
	}

	std::cout << "Separating channels\n";
	std::vector< std::vector< std::complex< float > > > input_waveform( channel_count );

	for( unsigned int i = 0; i <= training_samples.size(); ++i ) {
		unsigned int channel = i % channel_count;
		input_waveform[ channel ].push_back( std::complex< float >( training_samples[ i ], 0.f ) );
	}

	std::cout << "Performing fast fourier transform\n";

	std::vector< std::vector< std::vector< std::complex< float > > > > frequency_chunks( channel_count );

	for( unsigned int c = 0; c < channel_count; ++c ) {
		frequency_chunks[ c ] = stft( input_waveform[ c ], step_size );
	}

	std::cout << "Renormalizing\n";

	float max_amp = 0.f;

	for( unsigned int c = 0; c < channel_count; ++c ) {
		for( unsigned int i = 0; i < frequency_chunks[ c ].size(); ++i ) {
			for( unsigned int j = 0; j < frequency_chunks[ c ][ i ].size(); ++j ) {
				if( std::abs( frequency_chunks[ c ][ i ][ j ].real() ) > max_amp ) {
					max_amp = std::abs( frequency_chunks[ c ][ i ][ j ].real() );
				}

				if( std::abs( frequency_chunks[ c ][ i ][ j ].imag() ) > max_amp ) {
					max_amp = std::abs( frequency_chunks[ c ][ i ][ j ] );
				}
			}
		}
	}

	for( unsigned int c = 0; c < channel_count; ++c ) {
		for( unsigned int i = 0; i < frequency_chunks[ c ].size(); ++i ) {
			for( unsigned int j = 0; j < frequency_chunks[ c ][ i ].size(); ++j ) {
				frequency_chunks[ c ][ i ][ j ] /= std::complex< float >( max_amp, 0.f );
			}
		}
	}

	unsigned int epochs = 1;
	std::cout << "Enter number of epochs to train for: ";
	std::cin >> epochs;

	float mutability = 0.05f;
	std::cout << "Enter mutation rate: ";
	std::cin >> mutability;

	Vector input;
	input.setDimension( 0 );

	std::cout << "This may take a while...\n";

	for( unsigned int e = 0; e < epochs; ++e ) {
		std::cout << "Training epoch " << e << std::endl;
		network.resetState();

		for( unsigned int i = 0; i < frequency_chunks[ 0 ].size(); ++i ) {
			std::cout << i << '/' << frequency_chunks[ 0 ].size() << " chunks complete\n";

			Vector expected_sample;
			expected_sample.setDimension( step_size * channel_count * 2 );

			for( unsigned int j = 0; j < step_size; ++j ) {
				for( unsigned int c = 0; c < channel_count; ++c ) {
					unsigned int sample_pos = 2 * ( j * channel_count + c );
					expected_sample( sample_pos ) = frequency_chunks[ c ][ i ][ j ].real();
					expected_sample( sample_pos + 1 ) = frequency_chunks[ c ][ i ][ j ].imag();
				}
			}

			float loss = network.train( input, expected_sample, mutability );

			std::cout << "Loss on current sample = " << loss << std::endl;
		}
	}
}

int main() {
	std::cout << "Welcome to the audio-based Neural Network test - second attempt\n";

	std::cout << "Enter sample rate: ";
	std::cin >> sample_rate;

	std::cout << "Enter channel count (all training files must match this): ";
	std::cin >> channel_count;

	if( channel_count < 1 ) {
		channel_count = 1;
		std::cout << "Increased channel count to 1 (mono)\n";
	}

	if( channel_count > 2 ) {
		channel_count = 2;
		std::cout << "Decreased channel count to 2 (stereo)\n";
	}

	std::cout << "Enter STFT step size (enter 2048 if you do not know what this means): ";
	std::cin >> step_size;
	chunk_size = 2 * step_size;

	setupNetwork();

	bool running = true;
	while( running ) {
		char instruction;
		std::cout << "> ";
		std::cin >> instruction;

		switch( instruction ) {
			case 'g':
				instructGenerate();
				break;

			case 'h':
				instructHelp();
				break;

			case 'l':
				instructLoad();
				break;

			case 'q':
				std::cout << "Have a good day!\n";
				running = false;
				break;

			case 's':
				instructSave();
				break;

			case 't':
				instructTrain();
				break;

			default:
				break;
		}
	}

	return 0;
}
