#include "LinearAlgebra.hpp"
#include "NeuralNet.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <SFML/Audio/InputSoundFile.hpp>
#include <SFML/Audio/OutputSoundFile.hpp>

Vector memory( 0 );
unsigned int sample_rate;
unsigned int memory_samples;
std::unique_ptr< NeuralNetwork > network;

void instructGenerate();
void instructHelp();
void instructTrain();

int main() {
	std::cout << "Welcome to the audio-based neural network test!\n";

	if( lasettings::setupOpenCL() ) {
		std::cout << "Using OpenCL optimizations\n";
	} else {
		std::cout << "Not using OpenCL optimizations, program will run significantly slower\n";
	}

	unsigned int neuron_layers = 1;
	std::cout << "Enter desired number of hidden neuron layers: ";
	std::cin >> neuron_layers;

	unsigned int neuron_count = 100;
	std::cout << "Enter neurons per hidden layer: ";
	std::cin >> neuron_count;

	sample_rate = 48000;
	std::cout << "Enter the number of audio samples per second: ";
	std::cin >> sample_rate;

	memory_samples = 480;
	std::cout << "Enter network memory in samples: ";
	std::cin >> memory_samples;
	memory_samples *= 2; // Stereo

	network = std::make_unique< NeuralNetwork >( memory_samples, 2, neuron_layers, neuron_count );

	std::cout << "Use command 'h' for help\n";

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

			case 'q':
				std::cout << "Have a good day!\n";
				running = false;
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

void instructGenerate() {
	std::string output_filename;
	std::cout << "Give an output filename: ";
	std::cin >> output_filename;

	unsigned int length_samples;
	std::cout << "Give song length in seconds: ";
	std::cin >> length_samples;
	length_samples *= sample_rate;

	sf::OutputSoundFile output_file;
	if( !output_file.openFromFile( output_filename, sample_rate, 2 ) ) {
		std::cout << "Failed to open output file\n";
		return;
	}

	memory = Vector( memory_samples );
	std::vector< sf::Int16 > output_samples;

	for( unsigned int i = 0; i < length_samples; ++i ) {
		if( i % sample_rate == 0 ) {
			std::cout << i << '/' << length_samples << " samples rendered\n";
		}
		Vector sample = network->propagate( memory );
		output_samples.push_back( sample.at( 0 ) * 65535 - 36768 );
		output_samples.push_back( sample.at( 1 ) * 65535 - 36768 );
		memory.getInternalData().push_back( sample.at( 0 ) );
		memory.getInternalData().push_back( sample.at( 1 ) );
		memory.getInternalData().erase( memory.getInternalData().begin() );
		memory.getInternalData().erase( memory.getInternalData().begin() );
	}

	output_file.write( output_samples.data(), output_samples.size() );
}

void instructHelp() {
	std::cout << "List of commands:\n";
	std::cout << "g - Generate an output file\n";
	std::cout << "h - Print this help menu\n";
	std::cout << "q - Quit the application\n";
	std::cout << "t - Train on an audio file\n";
}

std::vector< float > readSamples( sf::InputSoundFile& input_file ) {
	std::vector< float > training_samples( 0 );
	sf::Int16 samples_in[ 1024 ];
	sf::Uint64 read_count = 0;

	do {
		read_count = input_file.read( samples_in, 1024 );
		for( unsigned int i = 0; i < read_count; ++i ) {
			training_samples.push_back( static_cast< float >( samples_in[ i ] + 32768 ) / 65535.f );
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

	if( training_file.getChannelCount() != 2 ) {
		std::cout << "Expected stereo sound file\n";
		return std::vector< float >( 0 );
	}

	return readSamples( training_file );
}

void instructTrain() {
	std::string training_filename;
	std::cout << "Enter filename of training file: ";
	std::cin >> training_filename;

	unsigned int epochs = 1;
	std::cout << "Enter number of epochs to train for: ";
	std::cin >> epochs;

	std::vector< float > training_samples_init = readTrainingFile( training_filename );

	if( training_samples_init.size() == 0 ) {
		return;
	}

	Vector training_samples( training_samples_init.size() );
	training_samples.getInternalData() = training_samples_init;

	memory = Vector( memory_samples );

	std::cout << "This may take a while...\n";

	for( unsigned int e = 0; e < epochs; ++e ) {
		std::cout << "Training epoch " << e << std::endl;
		for( unsigned int i = 0; i < training_samples.getLength() - 1; i += 2 ) {
			if( i % sample_rate == 0 ) {
				std::cout << i << '/' << training_samples.getLength() << " samples complete\n";
			}

			Vector expected_sample( 2 );
			expected_sample.at( 0 ) = training_samples.at( i );
			expected_sample.at( 1 ) = training_samples.at( i + 1 );

			network->backPropagate( memory, expected_sample, 0.05 );

			memory.getInternalData().push_back( training_samples.at( i ) );
			memory.getInternalData().push_back( training_samples.at( i + 1 ) );
			memory.getInternalData().erase( memory.getInternalData().begin() );
			memory.getInternalData().erase( memory.getInternalData().begin() );

			if( i % sample_rate == 0 ) {
				std::cout << "Loss on current sample: " << network->loss( memory, expected_sample ) << '\n';
			}
		}
	}
}
