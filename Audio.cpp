#include "NeuralNetwork.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <SFML/Audio/InputSoundFile.hpp>
#include <SFML/Audio/OutputSoundFile.hpp>

NeuralNetwork network;
std::vector< float > memory;
unsigned int sample_rate;

void instructGenerate();
void instructHelp();
void instructTrain();

int main() {
	std::cout << "Welcome to the audio-based neural network test!\n";

	unsigned int neuron_layers = 1;
	std::cout << "Enter desired number of hidden neuron layers: ";
	std::cin >> neuron_layers;

	network.setLayerCount( neuron_layers );

	unsigned int neuron_count = 100;
	std::cout << "Enter neurons per hidden layer: ";
	std::cin >> neuron_count;

	network.setHiddenNeuronCount( 100 );

	sample_rate = 48000;
	std::cout << "Enter the number of audio samples per second: ";
	std::cin >> sample_rate;

	float memory_length = 1.f;
	std::cout << "Enter network memory in seconds: ";
	std::cin >> memory_length;

	unsigned int memory_samples = std::ceil( 2 * memory_length * sample_rate );
	network.setInputCount( memory_samples );
	network.setOutputCount( 2 );

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
	if( !output_file.openFromFile( output_filename, 48000, 2 ) ) {
		std::cout << "Failed to open output file\n";
		return;
	}

	memory.clear();
	memory.resize( network.getInputCount(), 0.f );

	std::vector< sf::Int16 > output_samples;

	for( unsigned int i = 0; i < length_samples; ++i ) {
		std::vector< float > sample = network.propagate( memory );
		output_samples.push_back( sample[ 0 ] * 65535 - 36768 );
		output_samples.push_back( sample[ 1 ] * 65535 - 36768 );
		memory.push_back( sample[ 0 ] );
		memory.push_back( sample[ 1 ] );
		memory.erase( memory.begin() );
		memory.erase( memory.begin() );
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
	std::uint64_t sample_count = input_file.getSampleCount();
	std::vector< float > training_samples( sample_count );
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

	std::vector< float > training_samples = readTrainingFile( training_filename );

	if( training_samples.size() == 0 ) {
		return;
	}

	memory.clear();
	memory.resize( network.getInputCount(), 0.f );

	float loss = 0.f;

	unsigned int ten_percent = training_samples.size() / 10;

	std::cout << "This may take a while...\n";

	for( unsigned int i = 0; i < training_samples.size() - 1; i += 2 ) {
		std::vector< float > expected_sample;
		expected_sample.push_back( training_samples[ i ] );
		expected_sample.push_back( training_samples[ i + 1 ] );

		network.backPropagate( memory, expected_sample, 0.05 );
		loss += network.loss( memory, expected_sample );

		memory.push_back( training_samples[ i ] );
		memory.push_back( training_samples[ i + 1 ] );
		memory.erase( memory.begin() );
		memory.erase( memory.begin() );

		if( i % ten_percent == 0 ) {
			std::cout << i / ten_percent * 10 << "% complete\n";
		}
	}
}
