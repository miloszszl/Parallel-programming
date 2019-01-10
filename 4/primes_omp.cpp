#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <math.h>

using namespace std;

//klasa reprezentujaca liczbe wraz z informacja czy liczba jest pierwsza czy zlozona
class MyNumber{
    public:
        unsigned long long value;
        bool isPrime;
    MyNumber(unsigned long long value){
        this->value = value;    
    }
};

void validateNumOfArgs(int argc);
int getNumOfThreads(char* n);
vector<MyNumber> readNumbersFromFile(char* path);
bool isPrime(unsigned long long number);

int main (int argc, char** argv){
    //sprawdzanie czy przy wywolaniu programu podano odpowiednia ilosc argumentow
    validateNumOfArgs(argc);

    //pobranie liczby watkow
    int threads = getNumOfThreads(argv[1]);
    
    //odczyt liczb z pliku
    vector<MyNumber> numbersVector = readNumbersFromFile(argv[2]);
    
    double begin,end;
    unsigned int i;
    begin = omp_get_wtime();
    
    //sprawdzanie czy liczby sa pierwsze
    #pragma omp parallel for default(shared) private(i) schedule(dynamic) num_threads(threads)
    for (i=0;i<numbersVector.size();i++){
        numbersVector[i].isPrime = isPrime(numbersVector[i].value);
    }
    
    end = omp_get_wtime();
    
    //wypisanie czasu oraz wynikow obliczen
    cout<<"Time: "<<(end - begin) * 1000<<"ms"<<endl;
    for (i=0;i<numbersVector.size();i++){
        cout << numbersVector[i].value << ": " << (numbersVector[i].isPrime ? "prime":"composite") << endl;
        
    }

    return 0;
}

void validateNumOfArgs(int argc){
    //sprawdzenie czy liczba argumentow programu jest poprawna
    if(argc != 3){
        cout<<"Niepoprawna ilosc argumentow"<<endl;
        cout<<"Poprawne wywolanie: ./primes_omp <n> <primes>"<<endl;
        exit(-1);
    }
}

int getNumOfThreads(char* n){
    //konwersja ciagu znakow na liczbe calkowita z obsluga bledu
    int threads;
    try
    {
        threads = atoi(n);
        if(threads<1){
            throw invalid_argument("");
        }
        return threads;
    }catch(invalid_argument e)
    {
        cout <<  "Niepoprawna liczba watkow"<<endl;
        exit(-1);
    }
}

vector<MyNumber> readNumbersFromFile(char* path){
    //odczyt liczb z pliku i umieszczenie ich w kontenerze vector + obsluga bledow
    ifstream primesFile; 
    primesFile.open(path);

    vector<MyNumber> numbersVector;
    if (primesFile.is_open())
    {
        string line;
        while ( getline (primesFile,line) )
        {   
            if(!line.empty()){
                try{
                    //pobrana z pliku linia jest rzutowana na typ unsigned long long i umieszczana w kontenerze
                    numbersVector.push_back(MyNumber(stoull(line)));
                }catch(exception e){
                    continue;
                }
            }
        }
        primesFile.close();
        return numbersVector;
    }else{
        cout << "Nie mozna otworzyc pliku"<<endl;
        exit(-1);
    }
}

bool isPrime(unsigned long long number){
    //sprawdzanie czy liczba jest pierwsza
    unsigned long long i;
    unsigned long long numberSqrt = sqrt(number);

    if(number<2){
        return false;
    }

    for(i=2;i<=numberSqrt;i++){
        if(!(number%i)){
            return false;
        }
    }
    return true;
}

