#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <mpi.h>

using namespace std;

void validateNumOfArgs(int argc);
vector<unsigned long long> readNumbersFromFile(char* path);
unsigned long long modPow(unsigned long long a, unsigned int n, unsigned long long p);
bool isPseudoPrime(unsigned long long number, int iters = 100);

int main (int argc, char** argv){
    
	//inicjalizacja MPI
	MPI::Init(argc, argv);

    //pobranie numeru procesu oraz liczby procesow
	int rank, numOfTasks;
	rank = MPI::COMM_WORLD.Get_rank();
	numOfTasks = MPI::COMM_WORLD.Get_size();

    //wektor do przechowywania liczb odczytanych z pliku
    vector<unsigned long long> numbersFromFile;

    //wektor odpowiedzi czy liczby sa pierwsze czy zlozone
    vector<short> answers;

    //wektor okreslajacy liczby liczb do wyslania poszczegolnym procesom
    vector<int> sendcounts;

    //wektor okreslajacy przesuniecie wzgledem poczatku przesylanej struktury (potrzebny do wyslania liczb do procesow)
    vector<int> displs;

    //zmienna przechowujaca liczbe liczb otrzymanych przez kazdy proces od procesu o rank = 0
    int receivedDataSize;

    if(rank==0){

    	//sprawdzanie czy przy wywolaniu programu podano odpowiednia ilosc argumentow
    	validateNumOfArgs(argc);

    	//odczyt liczb z pliku
    	numbersFromFile = readNumbersFromFile(argv[1]);
    	answers.resize(numbersFromFile.size());

    	//wyznaczanie ilosci liczb na poszczegolny proces
    	int numbersPerProc = numbersFromFile.size() / numOfTasks;
    	int rest = numbersFromFile.size()-numbersPerProc*numOfTasks;

    	sendcounts = vector<int>(numOfTasks, numbersPerProc);
		for(int i=0;i<rest;i++){
			sendcounts[i]+=1;
		}
        
        //wyznaczanie przesuniec w wektorze danych do wyslania (potrzebne do scatterv)
        displs.push_back(0);
        for(int i=1;i<numOfTasks;i++){
            displs.push_back(displs[i-1] + sendcounts[i-1]);
        }

        //wysylanie(i odebranie) wartosci okreslajacej liczbe liczb do odebrania przez poszczegolne procesy
        //ta sama liczba okresla jak wiele odpowiedzi ma wyznaczyc kazdy proces, a nastepnie odeslac do procesu o rank = 0
    	MPI_Scatter(sendcounts.data(), 1, MPI_INT, &receivedDataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }else{
    	MPI_Scatter(NULL, 1, MPI_INT, &receivedDataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    //wektor do przechowywania odebranych liczb
    vector<unsigned long long> receivedNumbersVector(receivedDataSize);

    //wektor do przechowywania odpowiedzi czy liczby sa pierwsze czy zlozone 
    vector<short> partialAnswers(receivedDataSize);

    double begin,end;
    begin = MPI_Wtime();

    if(rank==0){
	   //wyslanie porcji liczb do procesow oraz odebranie porcji przez proces o rank = 0
    	MPI_Scatterv(numbersFromFile.data(), sendcounts.data(), displs.data(), MPI::UNSIGNED_LONG_LONG, receivedNumbersVector.data(), receivedDataSize, MPI::UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    }else{
	   //odebrani porcji liczb przez wszystkie procesy poza procesem o rank = 0
    	MPI_Scatterv(NULL, NULL, NULL, MPI::UNSIGNED_LONG_LONG, receivedNumbersVector.data(), receivedDataSize, MPI::UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    //sprawdzanie czy liczby sa pierwsze
    for (int i=0;i<receivedDataSize;i++){
    	partialAnswers[i] = isPseudoPrime(receivedNumbersVector[i]) ? 1:0;
    }

    //pobieranie danych od wszystkich procesow do porcesu 0
    if(rank==0){
    	MPI_Gatherv(partialAnswers.data(), receivedDataSize, MPI::SHORT, answers.data(),
			   sendcounts.data(), displs.data(), MPI::SHORT, 0, MPI_COMM_WORLD);
	}else{
    	MPI_Gatherv(partialAnswers.data(), receivedDataSize, MPI::SHORT, NULL,
			   NULL, NULL, MPI::SHORT, 0, MPI_COMM_WORLD);
    }

    end = MPI_Wtime();

    if(rank==0){
    	//wypisanie czasu
    	cout<<"Time: "<<(end - begin) * 1000<<"ms"<<endl;
        
        //wypisanie wynikow obliczen
        for(unsigned int i=0;i<numbersFromFile.size();i++){
            cout << numbersFromFile[i] << ": ";
			cout << ((answers[i]==1) ? "prime":"composite") << endl;
        }
    }

    //zakonczenie pracy srodowiska MPI
    MPI::Finalize();
    return 0;
}

void validateNumOfArgs(int argc){
    //sprawdzenie czy liczba argumentow programu jest poprawna
    if(argc != 2){
        cout<<"Niepoprawna ilosc argumentow"<<endl;
        cout<<"Poprawne wywolanie: mpirun -n <n> ./primes_mpi <primes>"<<endl;
        MPI_Abort(MPI::COMM_WORLD,-1);
    }
}

vector<unsigned long long> readNumbersFromFile(char* path){
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
        MPI_Abort(MPI::COMM_WORLD,-1);
    }
    //zwrocenie odczytanych z pliku liczb
    return numbersVector;
}

//potęgowanie modulo ( (a^n) mod p)
unsigned long long modPow(unsigned long long a, unsigned int n, unsigned long long p){

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
bool isPseudoPrime(unsigned long long p, int iters){

	if(p<2){			//liczby 0 i 1 nie sa pierwsze
		return false;
	}else if(p == 2){	//zabezpieczenie przed operacja mod 0
		return true;
	}

	int a;
	for(int i=0;i<iters;i++){
		//wybor a z przedzialu [2,p-1]
		a = rand()%(p-2) + 2;

		//sprawdzanie czy a^(p-1) = 1 (mod p)
		if(modPow(a, p-1, p) != 1){
			return false;	//liczba nie jest pierwsza
		}
	}

	return true;	//liczba jest prawdopodobnie pierwsza
}
