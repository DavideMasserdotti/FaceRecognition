import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Legge le immagini dal file zip
faces = {}
with zipfile.ZipFile("FaceDataset.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".pgm"):
            continue  # non è un immagine della faccia
        with facezip.open(filename) as image:
            # Se estraiamo i file da zip, possiamo invece usare cv2.imread(filename)
            faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)


# Visualizziamo delle facce usando matplotlib
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
faceimages = list(faces.values())[-16:]  # Prendiamo le ultime 16 immagini
for i in range(16):
    axes[i % 4][i // 4].imshow(faceimages[i], cmap="gray")
print("Showing sample faces")
plt.show()

# Stampiamo alcuni dettagli
faceshape = list(faces.values())[0].shape
print("Face image shape:", faceshape)

classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of images:", len(faces))

# Utilizziamo le classi da 1 a 39 per eigenface, non prendiamo tutta la classe 40 e
# l'immagione 10 della 39 per effettuare un test out-of-sample
facematrix = []
facelabel = []

for key, val in faces.items():
    if key == "s40/10.pgm":
        continue  # this is our test set
    if key.startswith("s39/"):
        continue  # this is our test set
    if key == "s38/10.pgm":
        continue  # this is our test set
    if key == "s37/10.pgm":
        continue  # this is our test set
    if key == "s36/10.pgm":
        continue  # this is our test set
    facematrix.append(val.flatten()) #array di array
    facelabel.append(key.split("/")[0])



# Definiamo una matrice NxM con N immagini e M pixel per immagine
facematrix = np.array(facematrix) #matrice

# Applichiamo PCA
pca = PCA().fit(facematrix)


#Prendiamo le prime K componenti principali come eigenface
n_components = 50
eigenfaces = pca.components_[:n_components]

# Mostriamo le prime 16 eigenface
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    axes[i % 4][i // 4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
print("Showing the eigenfaces")
plt.show()

# Generiamo la variabile weights come una matrice KxM dove K rappresenta il numero di eigenface e N il numero di campioni
weights = eigenfaces @ (facematrix - pca.mean_).T

print("Shape of the weight matrix:", weights.shape) # K = n.componenti principali, N = n. campioni

for i in range(36,41):
    # Test su un immagine out-of-sample di una nuova classe
    query = faces["s"+str(i)+"/10.pgm"].reshape(1, -1)
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    print("Best match %s with Euclidean distance %f" % (facelabel[best_match], euclidean_distance[best_match]))
    # Visualizziamo
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 6))
    axes[0].imshow(query.reshape(faceshape), cmap="gray")
    axes[0].set_title("Query")
    axes[1].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
    axes[1].set_title("Best match")
    plt.show()