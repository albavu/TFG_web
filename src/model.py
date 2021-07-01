### Cambio de la secuencia de texto a dinucleótidos
from re import X
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.models import load_model

def one_hot_encoder(my_array):
    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(['aa','ag','ac','at','tt','ta','tc','tg','gg','gt','ga','gc','cc','ca','cg','ct','nn']))
    integer_encoded = label_encoder.transform(my_array)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int,categories=[range(18)])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded

def getKmers(sequence, size=2):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


def resultado(secuencia):
    seq1 = getKmers(secuencia)
    seq1 = one_hot_encoder(seq1)
    seq1 = np.reshape(seq1, (1, 149,17))
    return seq1


def leer(archivo):
    total= pd.read_csv('/Users/alba_vu/Desktop/python_web/src/'+archivo.filename, delimiter=";")

    # function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
    def getKmers(sequence, size=2):
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
    
    total.columns = ['secuencia','nucleosoma']
    total.to_csv('/Users/alba_vu/Desktop/python_web/src/dinuc_total.csv')
    total['dinuc'] = total.apply(lambda x:getKmers(x['secuencia']),axis=1)
    total = total.drop('secuencia', axis=1)
    total = total.sample(frac=1).reset_index(drop=True)

    total.to_csv('/Users/alba_vu/Desktop/python_web/src/dinuc_total.csv')

    # function to convert a DNA sequence string to a numpy array
    # converts to lower case, changes any non 'acgt' characters to 'n'
    import re
    def string_to_array(my_string):
        my_string = my_string.lower()
        my_string = re.sub('[^acgt]', 'z', my_string)
        my_array = np.array(list(my_string))
        return my_array

    # create a label encoder with 'acgtn' alphabet
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(['aa','ag','ac','at','tt','ta','tc','tg','gg','gt','ga','gc','cc','ca','cg','ct','nn']))


    # function to one-hot encode a DNA sequence string
    # non 'acgt' bases (n) are 0000
    # returns a L x 17 numpy array
    def one_hot_encoder(my_array):
        integer_encoded = label_encoder.transform(my_array)
        onehot_encoder = OneHotEncoder(sparse=False, dtype=int,categories=[range(18)])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded = np.delete(onehot_encoded, -1, 1)
        return onehot_encoded

    
    # ## Reemplazo de dinucleotidos por z
    # Todos los dinucleótidos que contengan la n los sustituimos por z

    for a in range(total.shape[0]):
        for i in range(len(total.iloc[1,1])):
            if(total.iloc[a,1][i]=="na"):
                total.iloc[a,1][i] = "nn"
            elif(total.iloc[a,1][i]=="nc"):
                total.iloc[a,1][i] = "nn"
            elif(total.iloc[a,1][i]=="ng"):
                total.iloc[a,1][i] = "nn"
            elif(total.iloc[a,1][i]=="nt"):
                total.iloc[a,1][i] = "nn"
            elif(total.iloc[a,1][i]=="an"):
                total.iloc[a,1][i] = "nn"
            elif(total.iloc[a,1][i]=="cn"):
                total.iloc[a,1][i] = "nn"
            elif(total.iloc[a,1][i]=="gn"):
                total.iloc[a,1][i] = "nn"
            elif(total.iloc[a,1][i]=="tn"):
                total.iloc[a,1][i] = "nn"
        
    nueva2 = []
    for i in range(total.shape[0]):
        nueva2.append(one_hot_encoder(total.iloc[i,1]))

    import pickle
    with open("/Users/alba_vu/Desktop/python_web/src/dinuc_final.pickle","wb") as f:
        pickle.dump(nueva2,f)

def deepL():
    import pickle
    with open("/Users/alba_vu/Desktop/python_web/src/dinuc_final.pickle","rb") as f:
        data = pickle.load(f)

    x = np.array(data)
    clf=load_model('modeloDL_OHE_final_dinuc.h5')
    y_score = clf.predict(x)
    y_score = pd.DataFrame(y_score)
    # Aqui lo que hacemos es establecer un threshold para una categoría u otra (0-1)
    # No se puede meter en la matriz de confusión si no hacemos este paso prrevio

    #seq_predictions = list(map(lambda x: 0 if x<0.5 else 1, y_score.iloc[:,0]))
    #seq_predictions2 = list(map(lambda x: 0 if x<0.5 else 1, y_score.iloc[:,1]))
    seq_predictions = list(y_score.iloc[:,0])
    seq_predictions2 = list(y_score.iloc[:,1])
    seq_predictions = pd.DataFrame(seq_predictions)
    seq_predictions2 = pd.DataFrame(seq_predictions2)
    seq_predictions = pd.concat([seq_predictions,seq_predictions2],axis=1)
    seq_predictions.columns = ['0','1']
    return seq_predictions
