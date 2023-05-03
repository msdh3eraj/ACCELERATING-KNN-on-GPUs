#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRAIN_SIZE 100
#define TEST_SIZE 5000
#define IMAGE_SIZE 64
#define K 1

typedef struct {
    double *pixels;
    int label;
} Image;

double distance(Image img1, Image img2) {
    double dist = 0;
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        double diff = img1.pixels[i] - img2.pixels[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

int knn(Image test, Image dataset[], int num_images) {
    double distances[num_images];
    int indices[num_images];
    for (int i = 0; i < num_images; i++) {
        distances[i] = distance(test, dataset[i]);
        indices[i] = i;
    }
    for (int i = 0; i < num_images-1; i++) {
        for (int j = i+1; j < num_images; j++) {
            if (distances[j] < distances[i]) {
                double temp_dist = distances[i];
                int temp_index = indices[i];
                distances[i] = distances[j];
                indices[i] = indices[j];
                distances[j] = temp_dist;
                indices[j] = temp_index;
            }
        }
    }
    int counts[2] = {0, 0};
    for (int i = 0; i < K; i++) {
        int idx = indices[i];
        counts[dataset[idx].label]++;
    }
    return (counts[0] > counts[1]) ? 0 : 1;
}

void generate_images(Image images[], int num_images) {
   
    for (int i = 0; i < num_images; i++) {
        images[i].pixels = (double*) malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
        for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE; j++) {
            double pixel = (double) rand() / RAND_MAX;
            images[i].pixels[j] = pixel;
        }
        images[i].label = (rand() % 2);
    }
}

void free_images(Image images[], int num_images) {
    for (int i = 0; i < num_images; i++) {
        free(images[i].pixels);
    }
}

int main() {
    Image train_data[TRAIN_SIZE];
    Image test_data[TEST_SIZE];
    generate_images(train_data, TRAIN_SIZE);
    generate_images(test_data, TEST_SIZE);
    int num_correct = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        Image test_image = test_data[i];
        int predicted_label = knn(test_image, train_data, TRAIN_SIZE);
        if (predicted_label == test_image.label) {
            num_correct++;
        }
    }
    double accuracy = (double) num_correct / TEST_SIZE;
    printf("Accuracy: %f\n", accuracy);
    free_images(train_data, TRAIN_SIZE);
    free_images(test_data, TEST_SIZE);
    return 0;
}