#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstring>
#include <thread>
#include <chrono>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>

using namespace std;

int readGPIO(int num);
int writeGPIO(int num, std::string value);
int sendMessage(std::string message);

int main() {
//
	//FILE *file = fopen("/dev/gpio/2");

	//if (file == NULL) P

    /*while (true) {

    	//for (int x = 1; x < 30 ; x++) {
    	//	std::cout << readGPIO(x);
    	//}

    	cout << "LOOP" << endl;

    	writeGPIO(2, "0");
    	cout << readGPIO(2) << endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

    	writeGPIO(2, "1");
    	cout << readGPIO(2) << endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));




    }
    return 0;*/

	/*std::ifstream file("/dev/gpio/2", std::ios::in);

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
            sendMessage(line);
        } else {
        	//std::cout << "eof" << std::endl;
            file.clear(); // Clear EOF flag
            file.seekg(0, std::ios::beg);
        }
        //std::cout << "loop" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Adjust for real-time reading
    }

    std::cout << "exit" << std::endl;
    file.close();*/

	while (true) {

		int p1 = readGPIO(1);
		int p2 = readGPIO(2);
		int p3 = readGPIO(3);
		int p6 = readGPIO(6);

		std::cout << p1 << p2 << p3 << p6 << std::endl;

		std::ostringstream oss;
		oss  << p1 << p2 << p3 << p6;
		std::string s = oss.str();

		sendMessage(s);

		std::this_thread::sleep_for(std::chrono::milliseconds(100));

	}

    return 0;
}

int readGPIO(int num) {

	std::string gpio_path = "/dev/gpio/" + std::to_string(num);
	std::ifstream file(gpio_path, std::ios::in);

	int value = 2;

	if (file.is_open()) {

		file >> value;
		file.close();
		return value;

	} else {
		std::cerr << "Error Opening File: " << strerror(errno) << std::endl;
		return 2;
	}

}

int writeGPIO(int num, std::string value) {

	std::string gpio_path = "/dev/gpio/" + std::to_string(num);
	std::ofstream file(gpio_path, std::ios::in);

	if (file.is_open()) {

		file << value;
		file.close();
		return 0;

	} else {
		std::cerr << "Error Opening File: " << strerror(errno) << std::endl;
		return 1;
	}

}


int sendMessage(std::string message) {

	int sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock < 0) {
		std::cerr << "Socket fialed to create." << strerror(errno) << std::endl;

		return -1;
	}

	struct sockaddr_in serverAddr;
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_port = htons(12346);
	serverAddr.sin_addr.s_addr = inet_addr("192.168.2.96");

    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Connection failed!" << strerror(errno) << std::endl;
        return -1;
    }

    const char *msg = message.c_str();

    if (send(sock, msg, strlen(msg), 0) < 0) {
        std::cerr << "Send failed!" << std::endl;
        return -1;
    }

    std::cout << "Message sent to the server!" << std::endl;

    close(sock);

    return 0;

}
