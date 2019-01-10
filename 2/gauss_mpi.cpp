#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>
#include <math.h>

using namespace std;
using namespace cv;

#define maskDim 5

void validateNumOfArgs(int argc);
Mat readImage(char* location);
int sumMaskWeights(int mask[][maskDim]);
void blurImage(Mat& image,Mat& originalImage,int mask[][maskDim],int weightsSum);


int main(int argc, char** argv){
    
    //inicjalizacja maski
    int mask[maskDim][maskDim]={
        {1,1,2,1,1},
        {1,2,4,2,1},
        {2,4,8,4,2},
        {1,2,4,2,1},
        {1,1,2,1,1}    
    };

    //obliczenie sumy wag w masce
    int weightsSum = sumMaskWeights(mask);

    //inicjalizacja MPI
    MPI::Init(argc, argv);

    //pobranie numeru procesu oraz liczby procesow
    int rank, numOfTasks;
    rank = MPI::COMM_WORLD.Get_rank();
    numOfTasks = MPI::COMM_WORLD.Get_size();

    //tworzenie tablicy potrzebnej do przechowywania informacji o rozmiarach porcji przesylanych danych
    int* sendcounts = new int[numOfTasks];

    //tworzenie tablicy potrzebnej do przechowywania informacji o pozycjach w buforze, z ktorych beda przesylane dane 
    int* displs = new int[numOfTasks];

    //tworzenie tablicy potrzebnej do przechowywania informacji o rozmiarach porcji otrzymywanych danych
    int* rcvcounts = new int[numOfTasks];

    //tworzenie tablicy potrzebnej do przechowywania informacji o pozycjach w buforze, z ktorych dane beda odbierane
    int* rcvdispls = new int[numOfTasks];

    //tablica do przechowywania danych takich jak: wysokosc, szerokosc i rozmiar danych w bajtach, dla poszczegolnych procesow
    int* recivedSizeData = new int[3];

    //zbiorcza tablica, ktora zostanie podzielona i umieszczona w poszczegolnych procesach w zmiennej recivedSizeData
    int* helperData = new int[numOfTasks*3];

    //deklaracja zmiennej, do ktorej zostanie wczytany obraz
    Mat inImage;

    //kod wykonywany tylko przez pierwszy proces
    if(rank==0){
        //sprawdzanie czy liczba podanych arumentow jest wlasciwa
        validateNumOfArgs(argc);
       
        //wczytanie obrazu do zmiennej inImage
        inImage = readImage(argv[1]);

        //wyznaczenie liczby wierszy obrazu, dla poszczegolnych procesow
        int rowsPerProc,restOfRows;
        rowsPerProc = static_cast<int>(ceil(static_cast<double>(inImage.rows) / numOfTasks));
        //jesli liczba wierszy nie dzieli sie bez reszty przez ilosc procesow, to obliczana jest pozostala liczba wierszy
        restOfRows = inImage.rows%rowsPerProc;
 
        //rozmiar wiersza w bajtach
        int bytesInRow = inImage.cols * inImage.elemSize();
        
        //zmienna oznaczajaca nadmiar, ktory potrzeba przeslac po to by nie bylo poziomych paskow na zdjeciu wynikowym
        int additionalSize = 2 * bytesInRow;

        //ustalanie wartosci niezbednych do przeslania i odebrania danych
        displs[0] = rcvdispls[0] = 0;
        if(numOfTasks==1){
            sendcounts[0] = rowsPerProc * bytesInRow; 
            helperData[0] = rowsPerProc;
            rcvcounts[0] = rowsPerProc * bytesInRow;
            helperData[2] = rowsPerProc * bytesInRow;
            helperData[1] = inImage.cols;
        }else{
            for(int i=0;i<numOfTasks;i++){
                //ustalenie ilosci danych do przeslania i odebrania (w bajtach)
                //ustalenie miesc, z ktorych nalezy pobierac i wysylac okreslone ilosci danych
                //ustalenie ilosci wierszy i kolumn do przeslania i odebrania 
                if(i<numOfTasks-1 || (restOfRows==0 && i==(numOfTasks-1))){
                    if(i==0 || (i==numOfTasks-1)){
                        sendcounts[i] = rowsPerProc * bytesInRow + additionalSize; 
                        helperData[3*i] = rowsPerProc + 2;
                    }else{
                        sendcounts[i] = rowsPerProc * bytesInRow + additionalSize*2;
                        helperData[3*i] = rowsPerProc + 4;
                    }
                    rcvcounts[i] = rowsPerProc * bytesInRow;
                    helperData[3*i+2] = rowsPerProc * bytesInRow;
                }else{
                    sendcounts[i] = restOfRows * bytesInRow + additionalSize;
                    rcvcounts[i] = restOfRows * bytesInRow;
                    helperData[3*i+2] = restOfRows * bytesInRow;
                    helperData[3*i] = restOfRows + 2;
                }
                
                if(i>0){
                    displs[i] = rcvdispls[i-1] + rcvcounts[i-1] - additionalSize ;
                    rcvdispls[i] = rcvdispls[i-1] + rcvcounts[i-1];
                }
                helperData[3*i+1] = inImage.cols;
            }
        }
        
        //wyslanie wysokosci, szerokosci i rozmiaru porcji danych(w bajtach) do odeslania do poszczegolnych procesow i odebranie tych danych przez proces root
        MPI_Scatter(helperData, 3, MPI_INT, recivedSizeData, 3, MPI_INT, 0, MPI_COMM_WORLD);
    }else{
        //odebranie wysokosci, szerokosci i rozmiaru porcji danych do odeslania w kazdym z procesow poza rootem
        MPI_Scatter(NULL, 3, MPI_INT, recivedSizeData, 3, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    //inicjowanie bufora, ktory przechowa odebrana czesc obrazu
    int sizeOfData = recivedSizeData[0]*recivedSizeData[1] * 3;
    unsigned char* receivedImageData = new unsigned char[sizeOfData];
    
    if(rank==0){
        //wyslanie porcji obrazu do procesow oraz odebranie porcji przez proces roota
        MPI_Scatterv(inImage.data, sendcounts, displs, MPI_BYTE, receivedImageData, sizeOfData, MPI_BYTE, 0, MPI_COMM_WORLD);
    }else{
        //odebrani porcji obrazu przez wszystkie procesy poza rootem
        MPI_Scatterv(NULL, NULL, NULL, MPI_BYTE, receivedImageData, sizeOfData, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    
    //inicjalizacja zmiennej przechowujacej przydzielona procesowi czesc obrazu (podanie wymiarow obrazu oraz zawartosci typu unsigned char*)
    Mat originalImagePart = Mat(recivedSizeData[0], recivedSizeData[1], CV_8UC3, receivedImageData);
    
    //klonowanie przydzielonej czesci obrazu 
    Mat copyOfImagePart = originalImagePart.clone();
    
    //poczatek pomiaru czasu
    double begin,end;
    begin = MPI_Wtime();
    
    //wykonanie rozmycia Gaussa na klonie obrazu
    blurImage(copyOfImagePart,originalImagePart,mask,weightsSum);
    
    //koniec pomiaru czasu
    end=MPI_Wtime();
    
    if(rank==0){
        //wyslanie czesci obrazu roota oraz skladanie wszystkich otrzymanych czesci obrazu w calosc przez proces roota przy wysylaniu pominiete zostaly dwa ostatnie wiersze)
        MPI_Gatherv(copyOfImagePart(Range(0, copyOfImagePart.rows-2), Range(0, copyOfImagePart.cols )).data,
             recivedSizeData[2], MPI_BYTE, inImage.data, rcvcounts, rcvdispls, MPI_BYTE, 0, MPI_COMM_WORLD);
    }else{
        if(rank==(numOfTasks-1)){
            //wyslanie czesci obrazu przefiltrowanej przez ostatni proces (przy wysylaniu pominiete zostaly dwa pierwsze wiersze)
            MPI_Gatherv(copyOfImagePart(Range(2, copyOfImagePart.rows), Range(0, copyOfImagePart.cols )).data, 
                recivedSizeData[2], MPI_BYTE, NULL, NULL, NULL, MPI_BYTE, 0, MPI_COMM_WORLD);
        }else{
            //wyslanie czesci obrazu przefiltrowanej przez wszystkie procesy poza pierwszym i ostatnim (przy wysylaniu pominiete zostaly dwa pierwsze i ostatnie wiersze)
            MPI_Gatherv(copyOfImagePart(Range(2, copyOfImagePart.rows-2), Range(0, copyOfImagePart.cols )).data,
                 recivedSizeData[2], MPI_BYTE, NULL, NULL, NULL, MPI_BYTE, 0, MPI_COMM_WORLD);
        }
    }
    
    //konczenie pracy MPI, po tek komendzie nie mozna korzystac z funkcjonalnosci MPI
    MPI::Finalize();

    if(rank==0){
        //wypisanie czasu
        cout<<"Czas: "<<(end - begin) * 1000<<"ms"<<endl;

        //zapisanie rozmytego obrazu    
        imwrite(argv[2],inImage);
    }
    
    //zwalnianie pamieci
    delete[] sendcounts;
    delete[] displs;
    delete[] rcvcounts;
    delete[] rcvdispls;
    delete[] recivedSizeData;
    delete[] helperData;
    delete[] receivedImageData;
    
    return 0;
}

void validateNumOfArgs(int argc){
    //sprawdzanie czy liczba argumentow programu jest odpowiednia
    if(argc != 3){
        cout<<"Niepoprawna ilosc argumentow"<<endl;
        cout<<"Poprawne wywolanie: mpirun -n <liczba_procesow> ./gauss_mpi <input_image> <output_image>"<<endl;
        MPI_Abort(MPI::COMM_WORLD,-1);
    }
}

Mat readImage(char* location){
    //wczytanie obrazu z podanej lokacji
    Mat image = imread(location, CV_LOAD_IMAGE_COLOR);
    if(!image.data)
    {
        cout <<  "Nie mozna otworzyc pliku"<<endl;
        MPI_Abort(MPI::COMM_WORLD,-1);
    }
    return image;
}

int sumMaskWeights(int mask[][maskDim]){
    //sumowanie elementow maski
    int sum = 0;
    for(int i=0;i<maskDim;i++){
        for(int j=0;j<maskDim;j++){
            sum+=mask[i][j];
        }    
    }
    return sum;
}

inline void blurImage(Mat& image,Mat& originalImage,int mask[][maskDim],int weightsSum){
    int maskElement;
    uchar* pixel;
    int red,green,blue,i,j,k,m;

    //algorytm rozmywania obrazu
    for(i=2;i<originalImage.rows-2;i++){
        for(j=2;j<originalImage.cols-2;j++){

            //zerowanie skladowych rgb
            red=green=blue=0;
            for(k=0;k<maskDim;k++){
                for(m=0;m<maskDim;m++){
                    //pobranie elementu maski
                    maskElement = mask[k][m];

                    //pobranie pixela z obrazu
                    pixel = originalImage.ptr<uchar>(i+k-2,j+m-2);  
                    
                    //modyfikacja skladowych rgb poprzez przemnozenie ich przez element maski
                    blue +=  maskElement * pixel[0] ;
                    green +=  maskElement * pixel[1] ;
                    red +=  maskElement * pixel[2] ;
                }            
            }

            //podzielenie skladowych rgb przez sume wag w masce oraz
            //modyfikacja pixela na pozycji [i,j] w obrazie wyjsciowym 
            image.at<Vec3b>(i,j) = Vec3b(blue/weightsSum,green/weightsSum,red/weightsSum);
        }    
    }
}
