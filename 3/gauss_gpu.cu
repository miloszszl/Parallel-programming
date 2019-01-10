#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
#define MASK_DIM 5

__constant__ int d_mask[MASK_DIM][MASK_DIM];
__constant__ int d_maskWeight;

int BLOCK_DIM = 32;

void validateNumOfArgs(int argc);
Mat readImage(char* location);
int sumMaskWeights(int mask[][MASK_DIM]);
__global__ void blurImage(unsigned char* d_imageBlur,unsigned char* d_imageOriginal, int rows, int cols, int channels);

int main(int argc, char** argv){
    
	//walidacja ilosci argumentow programu
	validateNumOfArgs(argc);

    //inicjalizacja maski
	 int mask[MASK_DIM][MASK_DIM]={
		{1,1,2,1,1},
		{1,2,4,2,1},
		{2,4,8,4,2},
		{1,2,4,2,1},
		{1,1,2,1,1}
	};

    //obliczenie sumy wag w masce
    int maskWeight = sumMaskWeights(mask);

    int sizeInt = sizeof(int);
	int sizeMask = MASK_DIM*MASK_DIM*sizeInt;

    //kopiowanie maski do pamieci stalych
	cudaMemcpyToSymbol(d_mask,mask,sizeMask);
	//kopiowanie wagi maski do pamieci stalych
	cudaMemcpyToSymbol(d_maskWeight,&maskWeight,sizeInt);

    //ladowanie obrazu do zmiennej inImage
    Mat imageOriginal = readImage(argv[1]);

    unsigned char *d_imageOriginal, *d_imageBlur;

    //rozmiar obrazu w bajtach (rozmiar wiersza w bajtach * ilosc elementow w wierszu)
    int imageSizeBytes = imageOriginal.step[0] * imageOriginal.rows;

    //alokacja pamieci na karcie dla obrazow
    cudaMalloc(&d_imageOriginal,imageSizeBytes);
    cudaMalloc(&d_imageBlur,imageSizeBytes);

    //kopiowanie obrazow do pamieci karty
    cudaMemcpy(d_imageOriginal, imageOriginal.ptr(), imageSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imageBlur, imageOriginal.ptr(), imageSizeBytes, cudaMemcpyHostToDevice);

    //okreslanie ilosci blokow i watkow w bloku
    dim3 blocks((imageOriginal.cols+BLOCK_DIM-1)*imageOriginal.channels()/BLOCK_DIM,
    		(imageOriginal.rows+BLOCK_DIM-1)/BLOCK_DIM);
    dim3 threads(BLOCK_DIM, BLOCK_DIM);

    //utworzenie eventów do pomiaru czasu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //start pomiaru czasowego
    cudaEventRecord(start);

    //rozmycie Gaussa na GPU
    blurImage<<<blocks,threads>>>(d_imageBlur,d_imageOriginal,imageOriginal.rows,imageOriginal.cols,imageOriginal.channels());

    //stop pomiaru czasowego
    cudaEventRecord(stop);

    //sychronizacja pomiaru stop
    cudaEventSynchronize(stop);

    //obliczanie roznicy czasow
    float timeMS;
    cudaEventElapsedTime(&timeMS, start, stop);

    //wypisanie uzyskanego czasu
    cout<<"Czas: "<<timeMS<<"ms"<<endl;

    //skopowanie rozmytego obrazu z pamieci karty graficznej do pamieci ram
    cudaMemcpy(imageOriginal.ptr(), d_imageBlur, imageSizeBytes, cudaMemcpyDeviceToHost);

    //zapisanie obrazu do pliku
    imwrite(argv[2],imageOriginal);

    //zwalnianie pamieci na karcie
    cudaFree(d_imageOriginal);
    cudaFree(d_imageBlur);

    return 0;
}

void validateNumOfArgs(int argc){
    //sprawdzanie czy liczba argumentow programu jest odpowiednia
    if(argc != 3){
        cout<<"Niepoprawna ilosc argumentow"<<endl;
        cout<<"Poprawne wywolanie: ./gauss_gpu <input_image> <output_image>"<<endl;
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

int sumMaskWeights(int mask[][MASK_DIM]){
    //sumowanie elementow maski
    int sum = 0;
    for(int i=0;i<MASK_DIM;i++){
        for(int j=0;j<MASK_DIM;j++){
            sum+=mask[i][j];
        }    
    }
    return sum;
}

__global__
void blurImage(unsigned char* d_imageBlur,unsigned char* d_imageOriginal, int rows, int cols, int channels){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int rowSize = cols*channels;
	int offset = x + y * rowSize;

	int k, m, partialPixelValue = 0;

	//2 pixele odstepu z kazdej strony
	if(y>2 && y<(rows-2) && x>(2*channels) && x<((cols-2)*channels)){

		for(k=0;k<MASK_DIM;++k){
			for(m=0;m<MASK_DIM;++m){
				//obliczanie skladowej piksela
				partialPixelValue+=d_mask[k][m]*d_imageOriginal[offset+(k-2)*rowSize+(m-2)*channels];
			}
		}

		//pobranie pixela obrazu rozmywanego
		d_imageBlur[offset] = partialPixelValue/d_maskWeight;
	}
}
