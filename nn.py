from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt


"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
images, labels = get_mnist()#unosimo slike i lables
#images-shape(60000,784) lables- shape(60000.10)
#weights
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784)) #daje random vrednosti izmedju 0-1, (20,784)- 20-hidden layer 784-input layer
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))#20-hidden layer 10-output layer
#bioses 
b_i_h = np.zeros((20, 1))#inicijalizujemo da budu 0 na pocetku
b_h_o = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 3# 3 puta prolazimo kroz sve podatke
#training neural network
for epoch in range(epochs):
    for img, l in zip(images, labels):#uzimamo jedan po jedan element 
        img.shape += (1,)#pretvaramo u matricu
        l.shape += (1,)#pretvaramo u matricu
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img # @-matrix multiplication
        h = 1 / (1 + np.exp(-h_pre))#activaciona funkcija koristimo je zato sto mozda h_pre moze da bude mnogo velika npr 9.0 pa je vracamo na odgovarajucu vrendost 
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
#posle forward propagation functin trebamo da poredimno rezultate sa lables
        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)#racuna gresku
        nr_correct += int(np.argmax(o) == np.argmax(l))#proverava da li je mreza klasifikovala output tacno i ako je tacno povecavamo brojac za 1

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l# razlika izmedju output and label
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index] # uzimamo sliku koju smo odredili iznad
    plt.imshow(img.reshape(28, 28), cmap="Greys") # ekstraktujemo sliku i dodajemo je ka plot obj

    # radimo forward propagation step da dobijemo ouypuy values
    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Subscribe if its a {o.argmax()} :)") # postavljamo title of the plot to the number of the strongest activated neuron
    plt.show() # pokazujemo plot
