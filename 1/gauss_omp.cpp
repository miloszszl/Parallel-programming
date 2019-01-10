#include <iostream>
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
#define maskDim 5

void validateNumOfArgs(int argc);
Mat readImage(char* location);
int getNumOfThreads(char* n);
int sumMaskWeights(int mask[][maskDim]);
void blurImage(Mat& image,int threads,int mask[][maskDim],int weightsSum);

int main(int argc, char** argv){
    //sprawdzanie czy liczba podanych arumentow jest wlasciwa
    validateNumOfArgs(argc);

    //wczytanie obrazu do zmiennej inImage
    Mat inImage = readImage(argv[2]);

    //sklonowanie obrazu po to by wykonywac rozmycie gaussa na klonie
    Mat clonedImage = inImage.clone();

    //pobranie ilosci watkow i rzutowanie argumentu na typ int
    int threads = getNumOfThreads(argv[1]);

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

    //wykonanie rozmycia gaussa na klonie obrazu
    blurImage(clonedImage,threads,mask,weightsSum);
    
    //zapisanie rozmytego obrazu    
    imwrite(argv[3],clonedImage);

    return 0;
}

void validateNumOfArgs(int argc){
    //sprawdzanie czy liczba argumentow programu jest odpowiednia
    if(argc != 4){
        cout<<"Niepoprawna ilosc argumentow"<<endl;
        cout<<"Poprawne wywolanie: ./gauss_omp <n> <input_image> <output_image>"<<endl;
        exit(-1);
    }
}

Mat readImage(char* location){
    //wczytanie obrazu z podanej lokacji
    Mat image = imread(location, CV_LOAD_IMAGE_COLOR);
    if(!image.data)
    {
        cout <<  "Nie mozna otworzyc pliku"<<endl;
        exit(-1);
    }
    return image;
}

int getNumOfThreads(char* n){
    //konwersja napisu do liczby calkowitej z obsluga bledu
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

inline void blurImage(Mat& image,int threads,int mask[][maskDim],int weightsSum){
    double begin,end;
    int maskElement;
    uchar* pixel;
    int red,green,blue,i,j,k,m;

    //start czasu
    begin = omp_get_wtime();
    
    //algorytm rozmywania obrazu
    #pragma omp parallel for default(shared) private(i,j,k,m,maskElement,pixel,red,green,blue)\
    schedule(static) num_threads(threads)
    for(i=2;i<image.rows-2;i++){
        for(j=2;j<image.cols-2;j++){

            //zerowanie skladowych rgb
            red=green=blue=0;
            for(k=0;k<maskDim;k++){
                for(m=0;m<maskDim;m++){
                    //pobranie elementu maski
                    maskElement = mask[k][m];

                    //pobranie pixela z obrazu
                    pixel = image.ptr<uchar>(i+k-2,j+m-2);  
                    
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

    //stop czasu
    end = omp_get_wtime();

    //wypisanie czasu rozmywania obrazu w milisekundach
    cout<<"Czas: "<<(end - begin) * 1000<<"ms"<<endl;
}
