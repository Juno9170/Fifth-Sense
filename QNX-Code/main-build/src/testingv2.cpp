#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstring>
#include <thread>
#include <chrono>

using namespace std;

int main() {
//
	//FILE *file = fopen("/dev/gpio/2");

	//if (file == NULL) P

	std::ifstream file("/dev/gpio/2", std::ios::in);

	if (!file.is_open()) {
		std::cerr << "Error Opening File: " << strerror(errno) << std::endl;
		return 1;
	}

    file.seekg(0, std::ios::end); // Move to the end of the file

    std::string line;
    while (true) {
        if (std::getline(file, line)) { // Read new lines
        	//std::cout << "line" << std::endl;
            std::cout << line << std::endl;
        } else {
        	//std::cout << "eof" << std::endl;
            file.clear(); // Clear EOF flag
            file.seekg(0, std::ios::beg);
        }
        //std::cout << "loop" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Adjust for real-time reading
    }

    std::cout << "exit" << std::endl;
    file.close();
    return 0;
}
