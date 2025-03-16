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

int readGPIO(std::ifstream& file);
int writeGPIO(std::ofstream& file, std::string value);
int sendMessage(int sock, sockaddr_in serverAddr, std::string message);
void checkPinStatuses(int end);

int main(int argc, char *argv[]) {

	const char* ipaddr = NULL;

	if (argc > 1) {

		ipaddr = argv[1];
		std::cout << "IP: " << ipaddr << std::endl;

	} else {
		ipaddr = "192.168.2.104";
		std::cout << "Setting default IP: " << ipaddr << std::endl;
	}

	std::ifstream gpio_1("/dev/gpio/1", std::ios::in);
	std::ifstream gpio_2("/dev/gpio/2", std::ios::in);
	std::ifstream gpio_3("/dev/gpio/3", std::ios::in);
	std::ifstream gpio_6("/dev/gpio/6", std::ios::in);
	std::ifstream gpio_7("/dev/gpio/7", std::ios::in);
	std::ifstream gpio_8("/dev/gpio/8", std::ios::in);
	std::ifstream gpio_9("/dev/gpio/9", std::ios::in);
	std::ifstream gpio_10("/dev/gpio/10", std::ios::in);

	int sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock < 0) {
		std::cerr << "Socket failed to create." << strerror(errno) << std::endl;

		return -1;
	}

	struct sockaddr_in serverAddr;
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_port = htons(12346);
	serverAddr.sin_addr.s_addr = inet_addr("192.168.2.104");

    /*if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Connection failed!" << strerror(errno) << std::endl;
        return -1;
    }*/

	while (true) {

		int p1 = readGPIO(gpio_1);
		int p2 = readGPIO(gpio_2);
		int p3 = readGPIO(gpio_3);
		int p6 = readGPIO(gpio_6);
		int p7 = readGPIO(gpio_7);
		int p8 = readGPIO(gpio_8);
		int p9 = readGPIO(gpio_9);
		int p10 = readGPIO(gpio_10);

		std::cout << p1 << p2 << p3 << p6 << p7 << p8 << p9 << p10 << std::endl;

		std::ostringstream oss;
		oss  << p1 << p2 << p3 << p6;
		std::string s = oss.str();

		if (sendMessage(sock, serverAddr, s) < 0) {

			std::this_thread::sleep_for(std::chrono::seconds(3));

			sock = socket(AF_INET, SOCK_STREAM, 0);

			if (sock < 0) {
				std::cerr << "Socket failed to create. Error: " << strerror(errno) << std::endl;
				close(sock);
				continue;
			}

		    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {

		        std::cerr << "Connection failed. Error: " << strerror(errno) << std::endl;
		        close(sock);
		    }

		}

		std::this_thread::sleep_for(std::chrono::milliseconds(100));

	}

	gpio_1.close();
	gpio_2.close();
	gpio_3.close();
	gpio_6.close();
	gpio_7.close();
	gpio_8.close();
	gpio_9.close();
	gpio_10.close();
	close(sock);

    return 0;
}

int readGPIO(std::ifstream& file) {

	//std::string gpio_path = "/dev/gpio/" + std::to_string(num);
	//std::ifstream file(gpio_path, std::ios::in);

	int value = 2;

	if (file.is_open()) {

        file.seekg(0, std::ios::beg);
        file >> value;
        file.clear();

		return value;

	} else {
		std::cerr << "Error Opening File: " << strerror(errno) << std::endl;
		return value;
	}

}

/*
 * Doesnt work. Maybe does?
 */
int writeGPIO(std::ofstream& file, std::string value) {

	//std::string gpio_path = "/dev/gpio/" + std::to_string(num);
	//std::ofstream file(gpio_path, std::ios::in);

	if (file.is_open()) {

		file << value;
		file.close();
		return 0;

	} else {
		std::cerr << "Error Opening File: " << strerror(errno) << std::endl;
		return 1;
	}

}


int sendMessage(int sock, sockaddr_in serverAddr, std::string message) {


    const char *msg = message.c_str();

    if (send(sock, msg, strlen(msg), 0) < 0) {
        std::cerr << "Send failed!" << std::endl;
        return -1;
    }

    std::cout << "Message sent to the server!" << std::endl;

    return 0;

}

void checkPinStatuses(int end) {

	for (int x = 0; x < end ; x++) {
		std::ifstream gpio("/dev/gpio/" + std::to_string(x), std::ios::in);
		std::cout << readGPIO(gpio);
		gpio.close();
	}

}
