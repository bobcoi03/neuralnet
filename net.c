// neural network in c
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h> // Include the time header
#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.5
#define EPOCHS 10000

typedef struct Neuron {
    float value;
    float *weights;
    float bias;
} Neuron;

typedef struct Layer {
    Neuron *neurons;
    int num_neurons;
} Layer;

typedef struct MLP {
    Layer input_layer;
    Layer hidden_layer;
    Layer output_layer;
} MLP;

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    // sigmoid(x) * (1 - x_)
    return x * (1 - x);
}

MLP create_mlp() {
    MLP mlp;
    // Initialize input layer
    mlp.input_layer.num_neurons = INPUT_SIZE;
    mlp.input_layer.neurons = (Neuron*)malloc(INPUT_SIZE * sizeof(Neuron));
    for (int i = 0; i < INPUT_SIZE; i++) {
        mlp.input_layer.neurons[i].value = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    // Initialize hidden layer
    mlp.hidden_layer.num_neurons = HIDDEN_SIZE;
    mlp.hidden_layer.neurons = (Neuron*)malloc(HIDDEN_SIZE * sizeof(Neuron));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        mlp.hidden_layer.neurons[i].value = 0.0;
        mlp.hidden_layer.neurons[i].weights = (float*)malloc(INPUT_SIZE * sizeof(float));
        for (int j = 0; j < INPUT_SIZE; j++) {
            mlp.hidden_layer.neurons[i].weights[j] = ((float)rand() / RAND_MAX) * 2 - 1; // Random weights between -1 and 1
        }
        mlp.hidden_layer.neurons[i].bias = ((float)rand() / RAND_MAX) * 2 - 1; // Random bias between -1 and 1
    }
    // Initialize output layer
    mlp.output_layer.num_neurons = OUTPUT_SIZE;
    mlp.output_layer.neurons = (Neuron*)malloc(OUTPUT_SIZE * sizeof(Neuron));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        mlp.output_layer.neurons[i].value = 0.0;
        mlp.output_layer.neurons[i].weights = (float*)malloc(HIDDEN_SIZE * sizeof(float));
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            mlp.output_layer.neurons[i].weights[j] = ((float)rand() / RAND_MAX) * 2 - 1; // Random weights between -1 and 1
        }
        mlp.output_layer.neurons[i].bias = ((float)rand() / RAND_MAX) * 2 - 1; // Random bias between -1 and 1
    }
    return mlp;
}

void feedforward(MLP *mlp) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
	        mlp->hidden_layer.neurons[i].value += mlp->hidden_layer.neurons[i].weights[j] * mlp->input_layer.neurons[j].value;
	    }	    
        mlp->hidden_layer.neurons[i].value += mlp->hidden_layer.neurons[i].bias;
        mlp->hidden_layer.neurons[i].value = sigmoid(mlp->hidden_layer.neurons[i].value);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            mlp->output_layer.neurons[i].value += mlp->output_layer.neurons[i].weights[j] * mlp->hidden_layer.neurons[j].value;
        }
        mlp->output_layer.neurons[i].value += mlp->output_layer.neurons[i].bias;
        mlp->output_layer.neurons[i].value = sigmoid( mlp->output_layer.neurons[i].value);
    }
};

float squared_error(float value, float predict) {
    return (value - predict) * (value - predict);
}

float cross_entropy_loss(float value, float predict) {
    // L = -[y * log(p) + (1 - y) * log(1 - p)]
    // L is the loss // binary classification only
    float loss = -(value * log(predict) + (1 - value) * log(1 - predict));
    return loss;
}

void print_values(MLP *mlp) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        printf("v: %f ", mlp->input_layer.neurons[i].value);
    }
    printf("\n");
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            printf("w: %f ", mlp->hidden_layer.neurons[i].weights[j]);
        }
        printf("h: %f ", mlp->hidden_layer.neurons[i].value);
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            printf("w: %f ", mlp->output_layer.neurons[i].weights[j]);
        }
        printf("\n");
        printf("o v: %f ", mlp->output_layer.neurons[i].value);
    }
    printf("\n");
}

void backprop(MLP *mlp, float target) {
    float delta_outputs[OUTPUT_SIZE] = {};    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        delta_outputs[i] = mlp->output_layer.neurons[i].value - target;    
    }
    float delta_hidden[HIDDEN_SIZE] = {};
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum += delta_outputs[j] * mlp->output_layer.neurons[j].weights[i];
        }
        delta_hidden[i] = sum * sigmoid_derivative(mlp->hidden_layer.neurons[i].value);
    }
    // Update weights and biases for output layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            // Update weights between hidden and output layers
            mlp->output_layer.neurons[i].weights[j] = mlp->output_layer.neurons[i].weights[j] - LEARNING_RATE * mlp->hidden_layer.neurons[j].value * delta_outputs[i];
        }
        // Update bias for this output neuron
        mlp->output_layer.neurons[i].bias = mlp->output_layer.neurons[i].bias - LEARNING_RATE * delta_outputs[i];
    }
    // Update weights and biases for hidden layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            mlp->hidden_layer.neurons[i].weights[j] = mlp->hidden_layer.neurons[i].weights[j] - LEARNING_RATE * mlp->input_layer.neurons[j].value * delta_hidden[i];
        }
        mlp->hidden_layer.neurons[i].bias = mlp->hidden_layer.neurons[i].bias - LEARNING_RATE * delta_hidden[i];
    }
}

int main() {
    srand(time(NULL));
    MLP mlp = create_mlp();
    // XOR datasets
    float inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float targets[4] = {0, 1, 1, 0};
    FILE *loss_file = fopen("loss_values.txt", "w");
    if (loss_file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0;
        for (int i = 0; i < 4; i++) {
            // Set input values
            for (int j = 0; j < INPUT_SIZE; j++) {
                mlp.input_layer.neurons[j].value = inputs[i][j];
            }
            // Perform feedforward
            feedforward(&mlp);
            // Calculate loss
            float loss = cross_entropy_loss(targets[i], mlp.output_layer.neurons[0].value);
            total_loss += loss;
            // Perform backpropagation
            backprop(&mlp, targets[i]);
        }
        fprintf(loss_file, "%d %f\n", epoch, total_loss / 4);
        printf("Epoch %d, Loss: %f\n", epoch, total_loss / 4);
    }
    // Print final values after training
    fclose(loss_file);
    print_values(&mlp);
    // Free allocated memory
    return 0;
};
