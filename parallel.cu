#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRAIN_SIZE 100
#define TEST_SIZE 500
#define IMAGE_SIZE 64
#define K 1
#define CHUNK_SIZE (TEST_SIZE/32)

typedef struct {
    double *pixels;
    int label;
} Image;

__device__ double distanceBetweenImages(double *pixels1, double *pixels2) {
    double dist = 0;
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        double diff = pixels1[i] - pixels2[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

__global__ void knn_slave(Image *train_data, Image *test_data, int *results) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid * CHUNK_SIZE;
    int end = start + CHUNK_SIZE;
    if (tid == 31) {
        end = TEST_SIZE;
    }
    for (int i = start; i < end; i++) {
        double distances[TRAIN_SIZE];
        int indices[TRAIN_SIZE];
        for (int j = 0; j < TRAIN_SIZE; j++) {
            distances[j] = distanceBetweenImages(train_data[j].pixels, test_data[i].pixels);
            indices[j] = j;
        }
        for (int j = 0; j < TRAIN_SIZE - 1; j++) {
            for (int k = j + 1; k < TRAIN_SIZE; k++) {
                if (distances[k] < distances[j]) {
                    double temp_dist = distances[j];
                    int temp_index = indices[j];
                    distances[j] = distances[k];
                    indices[j] = indices[k];
                    distances[k] = temp_dist;
                    indices[k] = temp_index;
                }
            }
        }

        int counts[2] = {0, 0};
        for (int j = 0; j < K; j++) {
            int idx = indices[j];
            counts[train_data[idx].label]++;
        }
        results[i] = (counts[0] > counts[1]) ? 0 : 1;
    }
}


void generate_images(Image images[], int num_images) {
   
    for (int i = 0; i < num_images; i++) {
        cudaMallocManaged(&(images[i].pixels), IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
        for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE; j++) {
            double pixel = (double) rand() / RAND_MAX;
            images[i].pixels[j] = pixel;
        }
        images[i].label = (rand() % 2);
    }
}

void free_images(Image images[], int num_images) {
    for (int i = 0; i < num_images; i++) {
        cudaFree(images[i].pixels);
    }
}

int main() {
    Image *train_data, *test_data;
    int *results;
    cudaMallocManaged(&train_data, TRAIN_SIZE * sizeof(Image));
    cudaMallocManaged(&test_data, TEST_SIZE * sizeof(Image));
    cudaMallocManaged(&results, TEST_SIZE * sizeof(int));
    generate_images(train_data, TRAIN_SIZE);
    generate_images(test_data, TEST_SIZE);

    knn_slave<<<1,32>>>(train_data, test_data, results);
    cudaDeviceSynchronize();
    
    int num_correct = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        if (results[i] == test_data[i].label) {
            num_correct++;
        }
    }



    double accuracy = (double) num_correct / TEST_SIZE;
    printf("Accuracy: %f\n", accuracy);
    free_images(train_data, TRAIN_SIZE);
    free_images(test_data, TEST_SIZE);
    cudaFree(train_data);
    cudaFree(test_data);
    cudaFree(results);
    return 0;
}