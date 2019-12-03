from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout


class Classifier(Model):
    def __init__(self, network='resnet', num_classes=31):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        if network == 'resnet':
            self.feature_extractor = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3),
                                                pooling='avg')
        else:
            raise NotImplementedError(f"Network {network} is not implemented yet.")

        self.label_predictor = Dense(self.num_classes)

    def call(self, x, training, trim=0):
        feature = self.feature_extractor(x, training=training)
        if trim == 1:
            return self.label_predictor(feature, training=training), feature
        elif trim == 0:
            feature = self.label_predictor(feature, training=training)
            return feature, feature
        else:
            raise NotImplementedError(f"Trim {trim} is not implemented yet.")


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = Dense(1024, activation='relu')
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(1024, activation='relu')
        self.dropout2 = Dropout(0.5)
        self.dense3 = Dense(1, activation='sigmoid')

    def call(self, x, training):
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)
