
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
class Cifar10Loading:
    def __init__(self):
        (self.x_train , self.y_train) , (self.x_test , self.y_test) = tf.keras.datasets.cifar10.load_data() # type: ignore
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']


    def getShape(self):
        print('Shape of x_train is {}'.format(self.x_train.shape))
        print('Shape of x_test is {}'.format(self.x_test.shape))
        print('Shape of y_train is {}'.format(self.y_train.shape))
        print('Shape of y_test is {}'.format(self.y_test.shape))

    def setValidationData(self,split_size,random_state=0):
        self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(self.x_test, self.y_test, test_size = split_size, random_state = random_state)
        print('Shape of x_train is {}'.format(self.x_train.shape))
        print('Shape of x_test is {}'.format(self.x_test.shape))
        print('Shape of x_val is {}'.format(self.x_val.shape))
        print('Shape of y_train is {}'.format(self.y_train.shape))
        print('Shape of y_test is {}'.format(self.y_test.shape))
        print('Shape of y_val is {}'.format(self.y_val.shape))






class DataVisualization:

    def __init__(self) -> None:
        pass
        
    
    def get_samples(self,x_train):
        plt.figure(figsize=(8,8))
        for i in range(0,9):
            plt.subplot(330 + 1 + i)
            img = x_train[i]
            plt.imshow(img)
        plt.show()
    
    def classDitribution(self,data):
        y_train = data.y_train.flatten()
        y_test =  data.y_test.flatten()

        # Class names for CIFAR-10

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

        # Plot the distribution
        fig, axs = plt.subplots(1, 2, figsize=(15, 5)) 

        # Count plot for training set
        sns.countplot(x=y_train, ax=axs[0], palette="viridis")
        axs[0].set_title('Distribution of Training Data')
        axs[0].set_xlabel('Classes')
        axs[0].set_ylabel('Count')
        axs[0].set_xticks(range(10))
        axs[0].set_xticklabels(class_names, rotation=45)

        # Count plot for testing set
        sns.countplot(x=y_test, ax=axs[1], palette="viridis")
        axs[1].set_title('Distribution of Testing Data')
        axs[1].set_xlabel('Classes')
        axs[1].set_ylabel('Count')
        axs[1].set_xticks(range(10))
        axs[1].set_xticklabels(class_names, rotation=45)

        plt.tight_layout()
        plt.show()


class DataPreprocessing:
    def __init__(self) -> None:
        self.augmentor=ImageDataGenerator(
                            rotation_range=15,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,)

    def normalize(self,x):
        x = x.astype('float32')
        x = x/255.0
        return x
    
    def normalizeX_Data(self,data):
        data.x_train = self.normalize(data.x_train)
        data.x_test =  self.normalize(data.x_test)
        data.x_val =   self.normalize(data.x_val)
    

        
    



    


    
