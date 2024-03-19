import wandb

#Q2 Also see the initial trails. Commenting out so not to print everytime it runs

wandb.init(project='fashion_mnist_hyperparameter',entity='rrjadhao27',name='q1DL')

label_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
              5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

(train_X,train_Y),(test_X,test_Y)=fashion_mnist.load_data()

train_len=train_X.shape[0]

collected_items=[]

item_labels=[]

for i in range(train_len):

  if(label_map[train_Y[i]] not in item_labels):
    collected_items.append(train_X[i])

    item_labels.append(label_map[train_Y[i]])

  if (len(item_labels)==10):
    break


wandb.log({"Q1 Images": [wandb.Image(img, caption=lbl) for img,lbl in zip(collected_items,item_labels)]})





# (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data() # This data have 70k images of 28*28 in training set and 10k in test set(10% to keep aside)

# fig, axs = plt.subplots (5,2, figsize=(16,7))

# for i in range (5):
#   for j in range (2):
#     k = i*2 + j
#     axs[i,j].imshow(x_train[y_train==k][0],cmap = 'gray')



# #for i in range(10):
#     #plt.subplot(5,2,i+1)
#     #plt.imshow(x_train[i].reshape(28,28), cmap='gray')