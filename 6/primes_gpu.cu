#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

using namespace std;

void validateNumOfArgs(int argc);
vector<unsigned long long> readNumbersFromFile(char* path);
__device__ unsigned long long modPow(unsigned long long a, unsigned int n, unsigned long long p);
__global__ void checkPrimes(int size, bool* d_answers, unsigned long long* d_numbers, curandState* curandStates, int iters=100);
__global__ void setupCurand(curandState* states);

int THREADS = 1024;

int main (int argc, char** argv){

	//walidacja ilosci wprowadzonych argumentow
	validateNumOfArgs(argc);
	
    //pobranie liczb z pliku do tablicy
    vector<unsigned long long> numbersFromFile = readNumbersFromFile(argv[1]);

    //pobranie rozmiarow;
    int sizeNumbersBytes = numbersFromFile.size()*sizeof(unsigned long long);
	int sizeAnswersBytes = numbersFromFile.size()*sizeof(bool);

    //tablice liczb i odpowiedzi dla GPU
    unsigned long long* d_numbers;
    bool* d_answers;

    //alokacja pamieci na GPU
    cudaMalloc((void **)&d_numbers, sizeNumbersBytes);
	cudaMalloc((void **)&d_answers, sizeAnswersBytes);

	//kopoiwanie tablicy liczb do pamieci karty graficznej
	cudaMemcpy(d_numbers, numbersFromFile.data(), sizeNumbersBytes, cudaMemcpyHostToDevice);

	//okreslanie ilosci blokow
	int blocks = (numbersFromFile.size()+THREADS-1)/THREADS;

	//deklaracja oraz alokacja pamieci na statusy potrzebne do generacji liczb losowych
	curandState* d_states;
	cudaMalloc(&d_states, blocks*THREADS*sizeof(curandState));

	//inicjalizacja niezbedna do generacji liczb losowych
	setupCurand<<<blocks,THREADS>>>(d_states);

	//utworzenie eventów do pomiaru czasu
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//rozpoczecie pomiaru czasu
	cudaEventRecord(start);

	//sprawdzanie pierwszosci liczb
	checkPrimes<<<blocks,THREADS>>>(numbersFromFile.size(),d_answers,d_numbers,d_states);

	//zatrzymanie pomiaru czasu
	cudaEventRecord(stop);

	//sychronizacja pomiaru czasu
	cudaEventSynchronize(stop);

	//obliczanie roznicy czasow
	float elapsedTimeMS;
	cudaEventElapsedTime(&elapsedTimeMS, start, stop);

	//wypisanie czasu
	cout<<"Czas: "<<elapsedTimeMS<<"ms"<<endl;
	
	//tablica odpowiedzi dla CPU (alokacja pamieci)
	bool* answers = (bool*)malloc(sizeAnswersBytes);

	//kopiowanie odpowiedzi z pamieci na karcie do RAMu
	cudaMemcpy(answers, d_answers, sizeAnswersBytes, cudaMemcpyDeviceToHost);

	//wypisanie wynikow obliczen
	for(unsigned int i=0;i<numbersFromFile.size();i++){
		cout << numbersFromFile[i] << ": ";
		cout << ((answers[i]==1) ? "prime":"composite") << endl;
	}
    
	cudaFree(d_answers);
	cudaFree(d_numbers);
	free(answers);
    return 0;
}

void validateNumOfArgs(int argc){
    //sprawdzenie czy liczba argumentow programu jest poprawna
    if(argc != 2){
        cout<<"Niepoprawna ilosc argumentow"<<endl;
        cout<<"Poprawne wywolanie: ./primes_gpu <primes>"<<endl;
        exit(-1);
    }
}

vector<unsigned long long>  readNumbersFromFile(char* path){
    //odczyt liczb z pliku i umieszczenie ich w kontenerze vector + obsluga bledow
    ifstream primesFile; 
    primesFile.open(path);

    vector<unsigned long long> numbersVector;
    //otwarcie pliku do odczytu
    if (primesFile.is_open())
    {
        //wczytywanie liczb z pliku linia po linii
        string line;
        while ( getline (primesFile,line) )
        {   
            //jesli linia nie jest pusta to pobierana jest z niej liczba
            if(!line.empty()){
                try{
                    //pobrana z pliku linia jest rzutowana na typ unsigned long long i umieszczana w kontenerze
                    numbersVector.push_back(stoull(line));
                }catch(exception e){
                    continue;
                }
            }
        }
        //zamkniecie pliku
        primesFile.close();
    }else{
        //konczenie pracy wszystkich procesow
        cout << "Nie mozna otworzyc pliku"<<endl;
        exit(-1);
    }
    //zwrocenie odczytanych z pliku liczb
    return numbersVector;
}

//inicjalizacja curanda (do generacji liczb losowych)
__global__ void setupCurand(curandState* states){
  int tid = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1000, tid, 0, &states[tid]);
}

//potęgowanie modulo ( (a^n) mod p)
__device__ unsigned long long modPow(unsigned long long a, unsigned int n, unsigned long long p){

	//inicjalizacja zwracanego rezultatu
	unsigned long long result = 1;

	//wykonywanie operacji dopoki wykladnik jest wiekszy niz 0
	while(n>0){
		if(n%2 == 1){		//n jest nieprarzyste
			result = (result * a) % p;
		}

		// potegowanie modulo (a^2 mod p)
		a = (a*a) % p;
		//zmniejszenie wykladnika w taki sposob ze wykladnik = wykladnik/2
		n/=2;
	}

	return result;
}

//sprawdzanie czy liczba jest prawdopodobnie pierwsza
//uzyto testu pierwszosci Fermata
__global__ void checkPrimes(int size, bool* d_answers, unsigned long long* d_numbers, curandState* curandStates, int iters){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid<size){
		if(d_numbers[tid]<2){			//liczby 0 i 1 nie sa pierwsze
				d_answers[tid] = false;
				return;
			}else if(d_numbers[tid] == 2){	//zabezpieczenie przed operacja mod 0
				d_answers[tid] = true;
				return;
			}

			int a;
			float randomNumber;
			for(int i=0;i<iters;i++){
				//wybor a z przedzialu [2,p-1]
				randomNumber = curand_uniform(&curandStates[tid]);
				randomNumber *= (d_numbers[tid] - 2.000001);
				randomNumber += 2;
				a = (int)truncf(randomNumber);

				//sprawdzanie czy a^(p-1) = 1 (mod p)
				if(modPow(a, d_numbers[tid]-1, d_numbers[tid]) != 1){
					d_answers[tid] = false;		//liczba nie jest pierwsza
					return;
				}
			}

			d_answers[tid] =  true;	//liczba jest prawdopodobnie pierwsza
	}
}
